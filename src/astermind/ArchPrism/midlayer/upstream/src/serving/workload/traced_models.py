"""Adapted model forwards executed on lightweight symbolic operators only.

Qwen3 source: vLLM 0.30 model_executor/models/qwen3.py (attention/decoder)
and qwen2.py (Qwen2MLP, Qwen2Model). PP1, no auxiliary hidden states,
unquantized BF16, source vllm_c CUDA in-place Norm/RoPE profile. Constructors replace
weight loading/runtime modules; tensor .view is named .view_as_shape here.
V4 FFN/input preparation adapt vllm/models/deepseek_v4/nvidia/model.py and
attention.py for a declared synchronous modular, pure-text EP-off source.
GLM BF16/official FP8 adapts deepseek_v32/nvidia/model.py and attention.py, reusing the
DeepseekV2 MLP body and modular Runner frames for an explicit all-decode
scope, ordinary all-reduce/norm and EP1/EP8 with SP/MTP off; its optional target
chain shares the typed TP logits/greedy primitives.
K2.6 reuses the ordinary MLA wrapper and BF16 dense/shared modules; its
explicit CP1 all-decode FlashMLA source caller does not bind a GPU provider.
No torch import, real tensor allocation, CUDA access or numerical forward.
"""

from contextlib import contextmanager
import weakref

from src.kernel.memory_access import MemoryAllocation

from .model_config import ModelConfig, is_validated_qwen3_pipeline_model_scope, is_validated_qwen3_moe_ep2_model_scope
from dataclasses import replace

from .traced_forward import ForwardActivationGroup, ForwardCollectiveGroup, ForwardKVCacheGroup, ForwardPipelineGroup, ForwardTrace, ForwardWorkspace, ForwardSourceProgram, SourceProgramCache, ForwardCollectiveSourceProgram
from .traced_operators import (
    gqa_cache_shape, gqa_full_source_metadata,
    SymbolicAttention, SymbolicEmbedding, SymbolicGather, SymbolicGreedy,
    SymbolicLinear, SymbolicRowParallelLinear, SymbolicBatchedLinear, SymbolicDSANormRope, SymbolicDSAQuery,
    SymbolicConcat, SymbolicCopy, SymbolicDSAIndexer, SymbolicDSASparseAttention,
    SymbolicFloat32Cast, SymbolicBiasedSigmoidTopK, SymbolicPositionZeroMask,
    SymbolicMLACacheUpdate, SymbolicMLALatentDecode, SymbolicAttentionStateMerge,
    SymbolicGroupedRouting, SymbolicRoutedExperts,
    SymbolicSoftmaxTopK, SymbolicUnquantizedExperts, SymbolicExpertOwnerMask, unquantized_expert_source_metadata,
    SymbolicK3AttentionResidual, SymbolicK3SiTU, SymbolicK3Routing, SymbolicK3Experts, SymbolicK3Stage,
    SymbolicAllReduceRMSNorm,
    SymbolicLogitsRowSelect, SymbolicTPLogitsProcessor,
    SymbolicRMSNorm, SymbolicRotary, SymbolicSiluAndMul,
    SymbolicAdd, SymbolicScalarMultiply, SymbolicBlockFP8Linear, SymbolicV4GroupedOutput, SymbolicV4InverseRope,
    SymbolicV4QKVNorm, SymbolicV4RoutedExperts, SymbolicV4Routing,
    SymbolicV4Compressor, SymbolicV4KVInsert, SymbolicV4ShortTopK, SymbolicV4SparseAttention,
    SymbolicV4IndexerQuery, SymbolicV4ScoredTopK,
    SymbolicMHC, SymbolicVocabParallelEmbedding,
    _dsa_source_phase,
)
from .workload_graph import WorkloadKind
from .rope_dispatch import ROPE_PROGRAM_FUSED, vllm_rope_program


class QwenMLP:
    def __init__(self, trace, model, pool, prefix, *, group=None, collective=None, collective_inplace=False):
        tp_size = 1 if group is None else len(group.traces)
        if model.intermediate_size % tp_size:
            raise ValueError("dense intermediate width must divide the TP group")
        intermediate = model.intermediate_size // tp_size
        self.gate_up_proj = SymbolicLinear(trace, prefix + ":gate-up", pool,
            model.hidden_size, 2 * intermediate)
        self.act_fn = SymbolicSiluAndMul(trace, prefix + ":silu-mul", pool)
        self.down_proj = SymbolicLinear(trace, prefix + ":down", pool,
            intermediate, model.hidden_size)
        if group is not None:
            self.down_proj = SymbolicRowParallelLinear(self.down_proj, group, collective, inplace=collective_inplace)

    def __call__(self, x):
        # Qwen2MLP.forward, reused by Qwen3: preserve gate_up's local lifetime.
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def maybe_fused_act_quant(act_fn, x, linear):
    """Unquantized branch of vLLM's helper; never infer a quant producer.

    SymbolicLinear declares an unquantized weight/input contract. The source
    helper's get_input_quant_key is therefore None and returns act_fn(x).
    Quantized consumers require their own explicit fused/source program.
    """
    consumer = linear.projection if type(linear) is SymbolicRowParallelLinear else linear
    if type(consumer) is not SymbolicLinear:
        raise NotImplementedError("symbolic activation/quant fusion requires an explicit consumer quantization contract")
    return act_fn(x)


class LlamaMLP(QwenMLP):
    """Keep LlamaMLP.forward's overwritten-local lifetime, not Qwen's alias."""

    def __call__(self, x):
        x, _ = self.gate_up_proj(x)
        x = maybe_fused_act_quant(self.act_fn, x, self.down_proj)
        x, _ = self.down_proj(x)
        return x


class DeepseekV2MLP(QwenMLP):
    """The same source forward body, with TP-sharded dense/shared widths.

    The constructor declares whether down_proj owns TP reduction or the
    caller keeps its partial. Ordinary SiluAndMul has no V4 clamp.
    """

    def __init__(self, trace, model, pool, prefix, *, intermediate_size, tp_size, module_path,
                 reduce_results, reduction_group=None, reduction_config=None):
        if (model.compute_dtype != "bf16"
                or type(intermediate_size) is not int or intermediate_size <= 0
                or type(tp_size) is not int or tp_size <= 0 or intermediate_size % tp_size):
            raise NotImplementedError("symbolic DeepSeek dense/shared MLP requires whole BF16 TP shards")
        if type(reduce_results) is not bool:
            raise ValueError("MLP reduce_results must explicitly declare its source branch")
        if reduce_results:
            if (not isinstance(reduction_group, ForwardCollectiveGroup)
                    or len(reduction_group.traces) != tp_size):
                raise ValueError("MLP reduction must cover its complete TP group")
        elif reduction_group is not None or reduction_config is not None:
            raise ValueError("partial MLP output must not carry reduction metadata")
        local = intermediate_size // tp_size
        self.gate_up_proj = _source_linear(trace, model, pool, prefix + ":gate-up", model.hidden_size, 2 * local,
            module_path + ".gate_up_proj")
        self.act_fn = SymbolicSiluAndMul(trace, prefix + ":silu-mul", pool)
        self.down_proj = _source_linear(trace, model, pool, prefix + ":down", local, model.hidden_size,
            module_path + ".down_proj")
        if reduce_results:
            self.down_proj = SymbolicRowParallelLinear(self.down_proj, reduction_group, reduction_config)


class DeepseekV32MoERunner:
    """Declared BF16-compute modular Runner frames; output un-reduced.

    Fused/shared, monolithic, latent transforms, DP/SP, DBO and additional
    output scaling are outside this source profile. V32 passes
    apply_routed_scale_to_output=False, so the router applies its factor and
    the Runner receives1.0; no duplicate output scaling is inserted.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, resolution, invocation_id,
                 expert_parallel_size=1, expert_rank=0):
        if (model.model_type != "glm_moe_dsa" or type(tp_size) is not int
                or tp_size not in (1, 8) or model.num_shared_experts != 1):
            raise NotImplementedError("symbolic V32 Runner requires GLM TP1/TP8 with one shared expert")
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size, model.num_experts,
            kind=WorkloadKind.ROUTER, output_dtype="fp32")
        self.shared_experts = DeepseekV2MLP(trace, model, pool, prefix + ":shared", tp_size=tp_size,
            intermediate_size=model.moe_intermediate_size * model.num_shared_experts,
            module_path=f"model.layers.{layer_index}.mlp.shared_experts", reduce_results=False)
        self.router = SymbolicGroupedRouting(trace, prefix + ":route", pool, model)
        self.routed_experts = SymbolicRoutedExperts(trace, prefix + ":routed", pool, model, tp_size=tp_size,
            layer_index=layer_index,
            resolution=resolution, invocation_id=invocation_id, routing_source_stage="dsv32_moe_route",
            expert_parallel_size=expert_parallel_size, expert_rank=expert_rank)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)

    def _apply_quant_method(self, hidden_states, router_logits, shared_experts_input):
        shared_output = self.shared_experts(shared_experts_input)
        topk_weights, topk_ids = self.router(hidden_states, router_logits)
        fused_output = self.routed_experts(hidden_states, topk_weights, topk_ids)
        return shared_output, fused_output

    def _forward_impl(self, hidden_states, router_logits, shared_experts_input):
        router_logits, _ = self.gate(hidden_states)
        return self._apply_quant_method(hidden_states, router_logits, shared_experts_input)

    def __call__(self, *, hidden_states, router_logits):
        if router_logits is not hidden_states:
            raise ValueError("V32 Runner must own its source gate, given the hidden-state placeholder")
        shared_output, fused_output = self._forward_impl(hidden_states, router_logits, hidden_states)
        return self.add(shared_output, fused_output)


class DeepseekV32MoE:
    """DeepseekV2MoE.forward with sequence parallelism disabled."""

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, resolution, invocation_id,
                 expert_parallel_size=1, expert_rank=0):
        self.experts = DeepseekV32MoERunner(trace, model, pool, prefix,
            layer_index=layer_index, tp_size=tp_size, resolution=resolution, invocation_id=invocation_id,
            expert_parallel_size=expert_parallel_size, expert_rank=expert_rank)

    def __call__(self, hidden_states):
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view_as_shape(num_tokens, hidden_dim)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        return final_hidden_states.view_as_shape(num_tokens, hidden_dim)


class DeepseekV2KimiMoERunner(DeepseekV32MoERunner):
    """Declared modular/no-overlap K2.6 Runner, output-scale then late TP sum.

    DP/SP1 keeps the incoming token rows replicated. In the explicit EP8
    source branch, MoE TP is one and each rank owns a linear expert slice;
    the shared expert still uses the original TP8 shards. Neither routed
    output is pre-reduced, so the same late all-reduce combines both parts.
    """

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index, resolution, invocation_id,
                 expert_parallel_size=1):
        tp_size = 1 if group is None else len(group.traces)
        if (model.model_id != "moonshotai/Kimi-K2.6" or model.compute_dtype != "bf16"
                or model.weight_dtype != "mixed_int4_bf16" or tp_size not in (1, 8)
                or model.num_shared_experts != 1):
            raise NotImplementedError("Kimi Runner requires exact BF16-compute K2.6 TP1/TP8 with one separate shared expert")
        if (type(expert_parallel_size) is not int or expert_parallel_size not in (1, 8)
                or expert_parallel_size > tp_size or (group is None and config is not None)):
            raise NotImplementedError("Kimi Runner requires EP-off or its explicit replicated-token EP8 source")
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size, model.num_experts,
            kind=WorkloadKind.ROUTER)
        self.shared_experts = DeepseekV2MLP(trace, model, pool, prefix + ":shared", tp_size=tp_size,
            intermediate_size=model.moe_intermediate_size,
            module_path=f"language_model.model.layers.{layer_index}.mlp.shared_experts", reduce_results=False)
        self.router = SymbolicGroupedRouting(trace, prefix + ":route", pool, model, apply_routed_scale_to_output=True)
        self.routed_experts = SymbolicRoutedExperts(trace, prefix + ":routed", pool, model, tp_size=tp_size,
            layer_index=layer_index, resolution=resolution, invocation_id=invocation_id, routing_source_stage="kimi_moe_route",
            expert_parallel_size=expert_parallel_size,
            expert_rank=group.participants.index(trace.rank) if expert_parallel_size > 1 else 0)
        self.output_scale = SymbolicScalarMultiply(trace, prefix + ":output-scale", pool, model.routed_scaling_factor)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)
        self.group, self.config, self.name, self.pool = group, config, prefix, pool
        self.expert_parallel_size, self.invocation_id = expert_parallel_size, invocation_id

    def __call__(self, *, hidden_states, router_logits):
        if router_logits is not hidden_states:
            raise ValueError("Kimi Runner owns its BF16 gate and requires the hidden-state placeholder")
        shared_output, fused_output = self._forward_impl(hidden_states, router_logits, hidden_states)
        fused_output = self.output_scale(fused_output)
        result = self.add(shared_output, fused_output)
        if self.group is None:
            return result
        return self.group.call(self.name + ":all-reduce", result,
            pool_name=self.pool, config=self.config, inplace=False,
            moe_invocation_id=self.invocation_id if self.expert_parallel_size > 1 else None)


class DeepseekV2KimiMoE(DeepseekV32MoE):
    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index, resolution, invocation_id,
                 expert_parallel_size=1):
        self.experts = DeepseekV2KimiMoERunner(trace, model, pool, prefix, group, config,
            layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
            expert_parallel_size=expert_parallel_size)


def _source_linear(trace, model, pool, name, in_features, out_features, module_path):
    """Bind exact checkpoint exclusions to the shared linear primitives."""
    storage = model.weight_dtype_for_module(module_path, "Linear")
    if storage == "bf16":
        return SymbolicLinear(trace, name, pool, in_features, out_features)
    if storage == "fp8":
        return SymbolicBlockFP8Linear(trace, name, pool, in_features, out_features,
            source_profile="vllm_0_30_logical_group128_fp32")
    raise NotImplementedError("symbolic source linear requires BF16 or block-FP8 module weights")


class DeepseekV2MLAInput:
    """K2.6's ordinary MLA wrapper up to its attention call.

    Source: layers/mla.py MultiHeadLatentAttentionWrapper.forward, with the
    constructor's default fuse_qkv_rmsnorm=False, CP1 and no gates/indexer.
    The context retains the genuine caller locals while the backend runs.
    Attention/cache effects and the output projection are separate consumers.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, rotary_cache):
        if (model.model_type != "kimi_k2" or model.model_id != "moonshotai/Kimi-K2.6"
                or model.compute_dtype != "bf16" or model.target_semantics is None
                or model.target_semantics.rope_layout != "interleaved"
                or type(tp_size) is not int or tp_size not in (1, 8)):
            raise NotImplementedError("ordinary symbolic MLA input currently binds exact K2.6 BF16 TP1/TP8 CP1 source")
        if type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers:
            raise ValueError("MLA input layer must belong to the checkpoint")
        trace._check_tensor(rotary_cache)
        if (rotary_cache.dtype != "bf16" or len(rotary_cache.shape) != 2
                or rotary_cache.shape[-1] != model.qk_rope_head_dim):
            raise ValueError("MLA input requires its BF16 source rotary cache")
        self.trace, self.model, self.heads = trace, model, model.num_attention_heads // tp_size
        path = f"language_model.model.layers.{layer_index}.self_attn"
        self.fused_qkv_a_proj = _source_linear(trace, model, pool, prefix + ":qkv-a", model.hidden_size,
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim, path + ".fused_qkv_a_proj")
        self.q_a_layernorm = SymbolicRMSNorm(trace, prefix + ":q-a-norm", pool,
            model.q_lora_rank, model.target_semantics.rms_norm_eps)
        self.kv_a_layernorm = SymbolicRMSNorm(trace, prefix + ":kv-a-norm", pool,
            model.kv_lora_rank, model.target_semantics.rms_norm_eps)
        self.q_b_proj = _source_linear(trace, model, pool, prefix + ":q-b", model.q_lora_rank,
            self.heads * (model.qk_nope_head_dim + model.qk_rope_head_dim), path + ".q_b_proj")
        self.rotary_emb = SymbolicRotary(trace, prefix + ":rope", pool,
            self.heads, 1, model.qk_rope_head_dim, rotary_cache)

    @contextmanager
    def __call__(self, positions, hidden_states):
        self.trace._check_tensor(positions)
        self.trace._check_tensor(hidden_states)
        model, rows = self.model, hidden_states.shape[0]
        if (hidden_states.shape != (rows, model.hidden_size) or hidden_states.dtype != "bf16"
                or positions.shape != (rows,) or positions.dtype != "int64"):
            raise ValueError("MLA source inputs require BF16 hidden rows and matching INT64 positions")
        q_c = None
        kv_lora = None
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split((model.q_lora_rank, model.kv_lora_rank + model.qk_rope_head_dim))
        q_proj_layer = self.q_b_proj
        q_c = self.q_a_layernorm(q_c)
        q_proj_input = q_c
        kv_c, k_pe = kv_lora.split((model.kv_lora_rank, model.qk_rope_head_dim))
        kv_c_normed = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)
        q = q_proj_layer(q_proj_input)[0]
        q = q.view_as_shape(rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
        q_pe = q.narrow(-1, model.qk_nope_head_dim, model.qk_rope_head_dim)
        _, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        yield q, kv_c_normed, k_pe


class DeepseekV2MLADecodeAttention:
    """K2.6 TP8 CP1 all-decode wrapper, declared BF16 FlashMLA source.

    Library internals/compiled LSE program and physical page allocation remain
    unbound. The source operator program is never an installed provider.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, kv_lens, cache,
                 slot_mapping, slots, rotary_cache, reduction_group, reduction_config, backend_source, q_lens=None):
        if backend_source != "vllm_0_30_flashmla_bf16":
            raise NotImplementedError("MLA decode requires an explicit BF16 FlashMLA source declaration")
        if ((tp_size == 1 and (reduction_group is not None or reduction_config is not None))
                or (tp_size != 1 and (not isinstance(reduction_group, ForwardCollectiveGroup)
                    or len(reduction_group.traces) != tp_size))):
            raise ValueError("K2.6 attention requires its complete TP reduction group")
        self.trace, self.model, self.name = trace, model, prefix
        self.graph_token_shape = None
        self.inputs = DeepseekV2MLAInput(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=tp_size, rotary_cache=rotary_cache)
        self.heads = self.inputs.heads
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_mapping,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        self.q_absorb = SymbolicBatchedLinear(trace, prefix + ":q-absorb", pool,
            self.heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.concat = SymbolicConcat(trace, prefix + ":concat-query", pool)
        self.latent = SymbolicMLALatentDecode(trace, prefix + ":latent", pool, heads=self.heads,
            kv_lens=kv_lens, cache=cache, q_lens=q_lens,
            source_profile="vllm_0_30_causal_flashmla_bf16_source")
        if sum(self.latent.q_lens) != len(slots):
            raise ValueError("MLA decode needs one distinct update slot per query row")
        self.v_up = SymbolicBatchedLinear(trace, prefix + ":v-up", pool,
            self.heads, model.kv_lora_rank, model.v_head_dim)
        path = f"language_model.model.layers.{layer_index}.self_attn.o_proj"
        projection = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.heads * model.v_head_dim, model.hidden_size, path)
        self.o_proj = (projection if reduction_group is None
            else SymbolicRowParallelLinear(projection, reduction_group, reduction_config))

    def _forward_mqa(self, query, *, public_output_rows=None):
        query = self.concat(*query)
        return self.latent(query, public_output_rows=public_output_rows)

    def _mla(self, query, latent, rope):
        rows, model = query.shape[0], self.model
        active = sum(self.latent.q_lens)
        if rows != active:
            latent, rope = latent.narrow(0, 0, active), rope.narrow(0, 0, active)
        self.update(latent, rope)
        output = self.trace.tensor(self.trace.unique_name(self.name + ":attention-output"),
            (rows, self.heads * model.v_head_dim), "bf16")
        # FULL metadata's num_actual_tokens is itself padded. PIECEWISE
        # metadata keeps actual rows. The public backend output retains the
        # FULL extent, including seq-len-zero anonymous rows; useful cache
        # attention math still covers only the real request windows.
        if self.graph_token_shape is not None and self.graph_token_shape.runtime_mode == "FULL":
            return self._decode_into(query, output, public_output_rows=rows)
        if rows != active:
            self._decode_into(query.narrow(0, 0, active), output.narrow(0, 0, active))
            return output
        return self._decode_into(query, output)

    def _decode_into(self, query, output, *, public_output_rows=None):
        rows, model = query.shape[0], self.model
        q_nope, q_pe = query.split((model.qk_nope_head_dim, model.qk_rope_head_dim))
        absorbed = self.trace.tensor(self.trace.unique_name(self.name + ":absorbed-query"),
            (rows, self.heads, model.kv_lora_rank), "bf16")
        self.q_absorb(q_nope.transpose(0, 1), out=absorbed.transpose(0, 1))
        mqa_q = (absorbed, q_pe)
        attn_out, lse = self._forward_mqa(mqa_q, public_output_rows=public_output_rows)
        self.v_up(attn_out.transpose(0, 1),
            out=output.view_as_shape(rows, self.heads, model.v_head_dim).transpose(0, 1))
        return output

    def __call__(self, positions, hidden_states):
        self.trace._check_tensor(positions)
        self.trace._check_tensor(hidden_states)
        rows = sum(self.latent.q_lens)
        physical = rows if self.graph_token_shape is None else self.graph_token_shape.model_input_tokens
        if (hidden_states.shape != (physical, self.model.hidden_size) or hidden_states.dtype != "bf16"
                or positions.shape != (physical,) or positions.dtype != "int64"):
            raise ValueError("MLA decode requires exactly one BF16 hidden row and INT64 position per declared context")
        with self.inputs(positions, hidden_states) as (query, latent, rope):
            attn_out = self._mla(query, latent, rope)
            return self.o_proj(attn_out)[0]


class MLAFreshPrefillBackend:
    """Shared MLACommonImpl.forward_mha fresh BF16 source body.

    The prefill backend returns a fresh output (out=None); the caller copies
    it into its preallocated output. This does not bind a numerical provider,
    internal score/LSE workspace, physical traffic or a compiled launch.
    """

    def __init__(self, trace, model, pool, prefix, *, heads, q_lens, cache, module_path):
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.heads, self.q_lens, self.cache = heads, q_lens, cache
        self.kv_b_proj = _source_linear(trace, model, pool, prefix + ":kv-b", model.kv_lora_rank,
            heads * (model.qk_nope_head_dim + model.v_head_dim), module_path + ".kv_b_proj")
        self.copy_nope = SymbolicCopy(trace, prefix + ":key-nope-copy", pool)
        self.copy_rope = SymbolicCopy(trace, prefix + ":key-rope-copy", pool)
        self.copy_output = SymbolicCopy(trace, prefix + ":output-copy", pool)

    def __call__(self, query, latent, rope, output):
        model, rows = self.model, sum(self.q_lens)
        for value in (query, latent, rope, output):
            self.trace._check_tensor(value)
        if (any(value.dtype != "bf16" for value in (query, latent, rope, output))
                or query.shape != (rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
                or latent.shape != (rows, model.kv_lora_rank) or rope.shape != (rows, 1, model.qk_rope_head_dim)
                or output.shape != (rows, self.heads * model.v_head_dim)):
            raise ValueError("fresh MLA backend requires matching BF16 query, latent, RoPE and output rows")
        key, value = self._expand_kv(latent, rope)
        output_prefill = self._attention(query, key, value, kv_lens=self.q_lens,
            causal=True, return_lse=False)
        self.copy_output(output_prefill.view_as_shape(rows, self.heads * model.v_head_dim), output)

    def _expand_kv(self, latent, rope):
        model, rows = self.model, latent.shape[0]
        kv_nope = self.kv_b_proj(latent)[0].view_as_shape(
            rows, self.heads, model.qk_nope_head_dim + model.v_head_dim)
        k_nope, value = kv_nope.split((model.qk_nope_head_dim, model.v_head_dim))
        key = self.trace.tensor(self.trace.unique_name(self.prefix + ":key"),
            (rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim), "bf16")
        self.copy_nope(k_nope, key.narrow(-1, 0, model.qk_nope_head_dim))
        self.copy_rope(rope.expand(rows, self.heads, model.qk_rope_head_dim),
            key.narrow(-1, model.qk_nope_head_dim, model.qk_rope_head_dim))
        return key, value

    def _attention(self, query, key, value, *, kv_lens, causal, return_lse, lse_layout="heads_rows", allow_empty_kv=False):
        from .workload_graph import WorkloadNode
        model, rows = self.model, sum(self.q_lens)
        if lse_layout not in ("heads_rows", "rows_heads"):
            raise ValueError("MLA prefill LSE requires an explicit head/row source layout")
        if type(allow_empty_kv) is not bool or (allow_empty_kv and (causal or not return_lse)):
            raise ValueError("empty KV source requires noncausal context with explicit LSE")
        empty_context = allow_empty_kv and any(length == 0 for length in kv_lens)
        node = WorkloadNode(self.trace.unique_name(self.prefix + ":prefill-attention"), WorkloadKind.FLASH_ATTN,
            self.pool, {"batch": len(self.q_lens), "q_len": max(self.q_lens), "kv_len": max(kv_lens),
                "num_query_heads": self.heads, "qk_head_dim": model.qk_nope_head_dim + model.qk_rope_head_dim,
                "v_head_dim": model.v_head_dim, "kv_lora_rank": model.kv_lora_rank},
            dtype="bf16", variant="mla_prefill", rank=self.trace.rank, attributes={
                "source_component": ("symbolic_mla_context_prefill_mathematical" if return_lse
                    else "symbolic_mla_fresh_prefill_mathematical"), "launch_policy": "analytical_launch",
                "causal": causal, "kv_dist": "uniform", "kv_layout": "mla", "q_lens": self.q_lens,
                "kv_lens": kv_lens, "full_cuda_graphs": False, "max_cudagraph_capture_size": 0,
                "return_softmax_lse": return_lse, "mathematical_program": (
                    "bf16_mla_prefill_softmax_v1" if causal else
                        "bf16_mla_context_empty_rows_softmax_v1" if empty_context else "bf16_mla_context_softmax_v1"),
                "symbolic_workspace_complete": False})
        if return_lse:
            return self.trace.call_outputs(node, (query, key, value), outputs=(
                ("output", (rows, self.heads, model.v_head_dim), "bf16"),
                ("lse", (self.heads, rows) if lse_layout == "heads_rows" else (rows, self.heads), "fp32")),
                controls=(self.cache,))
        return self.trace.call(node, (query, key, value),
            shape=(rows, self.heads, model.v_head_dim), dtype="bf16", controls=(self.cache,))


class MLAContextPrefillBackend(MLAFreshPrefillBackend):
    """Nonempty contexts and single-request continuations in a logical arena.

    Mirrors forward_mha: causal suffix O/LSE, BF16 cached-latent gather and
    expansion, noncausal prefix O/LSE, then the final state merge. This is
    not a native block-table, compiler workspace or provider binding.
    """

    def __init__(self, *args, context_lengths, context_offsets, context_chunk_rows=None, **kwargs):
        super().__init__(*args, **kwargs)
        if (not isinstance(context_lengths, tuple) or not isinstance(context_offsets, tuple)
                or len(context_lengths) != len(self.q_lens) or len(context_offsets) != len(context_lengths)
                or any(type(length) is not int or length <= 0 for length in context_lengths)
                or any(type(offset) is not int or offset < 0 for offset in context_offsets)
                or any(offset + length > self.cache.shape[0]
                    for offset, length in zip(context_offsets, context_lengths, strict=True))):
            raise ValueError("MLA context requires positive request contexts and disjoint logical cache slices")
        intervals = sorted((offset, offset + length)
            for offset, length in zip(context_offsets, context_lengths, strict=True))
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("MLA context requires disjoint logical cache slices")
        self.context_lengths, self.context_offsets = context_lengths, context_offsets
        if context_chunk_rows is not None and (type(context_chunk_rows) is not int or context_chunk_rows <= 0):
            raise ValueError("MLA context chunk row budget must be a positive declared source value")
        if context_chunk_rows is not None and sum(context_lengths) > context_chunk_rows and len(context_lengths) != 1:
            raise NotImplementedError("multi-chunk MLA currently requires one nonempty context request")
        self.chunk_rows = sum(context_lengths) if context_chunk_rows is None else context_chunk_rows
        self.gather = SymbolicCopy(self.trace, self.prefix + ":context-cache-gather", self.pool)
        self.merge = SymbolicAttentionStateMerge(self.trace, self.prefix + ":context-merge", self.pool)
        self.context_accumulate = SymbolicAttentionStateMerge(self.trace, self.prefix + ":context-accumulate", self.pool)
        self.init_output = SymbolicCopy(self.trace, self.prefix + ":context-init-output", self.pool)
        self.init_lse = SymbolicCopy(self.trace, self.prefix + ":context-init-lse", self.pool)

    def __call__(self, query, latent, rope, output):
        rows, model = sum(self.q_lens), self.model
        if (query.shape != (rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
                or latent.shape != (rows, model.kv_lora_rank) or rope.shape != (rows, 1, model.qk_rope_head_dim)
                or output.shape != (rows, self.heads * model.v_head_dim)
                or any(value.dtype != "bf16" for value in (query, latent, rope, output))):
            raise ValueError("MLA context ports require matching BF16 rows")
        key, value = self._expand_kv(latent, rope)
        suffix, suffix_lse = self._attention(query, key, value, kv_lens=self.q_lens, causal=True, return_lse=True)
        # Source gather's typed BF16 payload in the declared logical arena.
        # Native page-table/starts metadata is unbound, not invented INT64
        # per-token row-index traffic.
        context_rows = sum(self.context_lengths)
        chunks = (((self.context_offsets[0] + start, min(self.chunk_rows, context_rows - start)),)
            for start in range(0, context_rows, self.chunk_rows)) if context_rows > self.chunk_rows else (
                tuple(zip(self.context_offsets, self.context_lengths, strict=True)),)
        workspace_rows = min(context_rows, self.chunk_rows)
        gathered = self.trace.tensor(self.trace.unique_name(self.prefix + ":context-workspace"),
            (workspace_rows, model.kv_lora_rank + model.qk_rope_head_dim), "bf16")
        # Keep one source gather and one expansion per planned chunk, with
        # explicit logical views instead of native block tables or row IDs.
        prefix, prefix_lse = None, None
        for chunk_index, chunk in enumerate(chunks):
            chunk_lengths = tuple(length for _, length in chunk)
            chunk_rows = sum(chunk_lengths)
            active = gathered if chunk_rows == workspace_rows else gathered.narrow(0, 0, chunk_rows)
            sources = tuple(self.cache.narrow(0, offset, length) for offset, length in chunk)
            node = self.gather.node(WorkloadKind.GATHER,
                {"elements": chunk_rows * gathered.shape[1], "rows": chunk_rows, "hidden": gathered.shape[1]},
                "bf16", "gather", {"gather_factor": 1, "index_count": 0,
                    "index_bytes": 8, "scattered": False, "materialized_output": True})
            self.trace.call(node, sources, shape=active.shape, dtype="bf16", out=active)
            context_latent, context_rope = active.split((model.kv_lora_rank, model.qk_rope_head_dim))
            key, value = self._expand_kv(context_latent,
                context_rope.view_as_shape(chunk_rows, 1, model.qk_rope_head_dim))
            partial, partial_lse = self._attention(query, key, value,
                kv_lens=chunk_lengths, causal=False, return_lse=True)
            if context_rows <= self.chunk_rows:
                prefix, prefix_lse = partial, partial_lse
            elif chunk_index == 0:
                # _compute_prefill_context initializes independent running
                # O/LSE after its first partial; only continuations merge.
                prefix = self.trace.tensor(self.trace.unique_name(self.prefix + ":context-output"),
                    partial.shape, "bf16")
                prefix_lse = self.trace.tensor(self.trace.unique_name(self.prefix + ":context-lse"),
                    partial_lse.shape, "fp32")
                self.init_output(partial, prefix)
                lse_copy = self.init_lse.node(WorkloadKind.GATHER,
                    {"elements": self.heads * rows, "rows": self.heads, "hidden": rows}, "fp32", "gather",
                    {"gather_factor": 1, "index_count": 0, "index_bytes": 8, "scattered": False,
                        "materialized_output": True, "mathematical_program": "materialized_fp32_copy_bytes_v1"})
                self.trace.call(lse_copy, (partial_lse,), shape=prefix_lse.shape, dtype="fp32", out=prefix_lse)
            else:
                self.context_accumulate(prefix, prefix_lse, partial, partial_lse,
                    out=prefix, out_lse=prefix_lse, inplace=True)
        self.merge(prefix, prefix_lse, suffix, suffix_lse,
            out=output.view_as_shape(rows, self.heads, model.v_head_dim))

class DeepseekV2MLAFreshPrefillAttention:
    """Ordinary BF16 MLA prefill, with an optional complete logical context.

    MLACommonPrefill.forward_mha passes out=None with unquantized output,
    then copies the backend's fresh output into the wrapper's output slice.
    This mathematical source domain does not bind a prefill provider.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, q_lens, cache,
                 slot_mapping, slots, rotary_cache, reduction_group, reduction_config,
                 context_length=0, context_slots=(), context_lengths=(), context_offsets=(), context_chunk_rows=None):
        if (not isinstance(q_lens, tuple) or not q_lens
                or any(type(length) is not int or length < 1 for length in q_lens)
                or sum(q_lens) != len(slots)):
            raise ValueError("fresh MLA requires one declared length per sequence and all update slots")
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.q_lens, self.cache = q_lens, cache
        self.inputs = DeepseekV2MLAInput(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=tp_size, rotary_cache=rotary_cache)
        self.heads = self.inputs.heads
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_mapping,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        path = f"language_model.model.layers.{layer_index}.self_attn"
        if context_length:
            if (context_lengths or context_offsets or context_slots != tuple(range(context_length))):
                raise ValueError("single-context compatibility ports require the explicit zero-based logical slice")
            context_lengths, context_offsets = (context_length,), (0,)
        elif context_slots:
            raise ValueError("context slots require a declared context length")
        backend = MLAContextPrefillBackend if context_lengths else MLAFreshPrefillBackend
        context = {"context_lengths": context_lengths, "context_offsets": context_offsets,
            "context_chunk_rows": context_chunk_rows} if context_lengths else {}
        if not context_lengths and context_offsets:
            raise ValueError("context offsets require matching positive context lengths")
        self.prefill_backend = backend(trace, model, pool, prefix,
            heads=self.heads, q_lens=q_lens, cache=cache, module_path=path, **context)
        projection = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.heads * model.v_head_dim, model.hidden_size, path + ".o_proj")
        self.o_proj = (projection if reduction_group is None
            else SymbolicRowParallelLinear(projection, reduction_group, reduction_config))

    def _forward_mha(self, query, latent, rope, output):
        self.prefill_backend(query, latent, rope, output)

    def __call__(self, positions, hidden_states):
        rows = sum(self.q_lens)
        if hidden_states.shape != (rows, self.model.hidden_size) or positions.shape != (rows,):
            raise ValueError("fresh MLA inputs must cover all declared sequence rows")
        with self.inputs(positions, hidden_states) as (query, latent, rope):
            self.update(latent, rope)
            output = self.trace.tensor(self.trace.unique_name(self.prefix + ":attention-output"),
                (rows, self.heads * self.model.v_head_dim), "bf16")
            self._forward_mha(query, latent, rope, output)
            return self.o_proj(output)[0]


class DeepseekV2MLAMixedAttention:
    """Decode-prefix plus fresh or cached-context prefill suffix."""

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, q_lens, kv_lens,
                 num_decodes, cache, slot_mapping, slots, rotary_cache, reduction_group, reduction_config,
                 context_chunk_rows=None):
        if (type(num_decodes) is not int or not 0 <= num_decodes < len(q_lens)
                or len(kv_lens) != len(q_lens)
                or (num_decodes and len(set(q_lens[:num_decodes])) != 1)
                or any(kv < q for q, kv in zip(q_lens, kv_lens))
                or (q_lens[num_decodes:] != kv_lens[num_decodes:]
                    and any(kv == q for q, kv in zip(q_lens[num_decodes:], kv_lens[num_decodes:])))):
            raise ValueError("MLA mixed source requires decode-first with fresh or nonempty-context prefill suffix")
        self.trace, self.model, self.prefix = trace, model, prefix
        self.rows, self.num_decodes = sum(q_lens), num_decodes
        self.decode_rows = sum(q_lens[:num_decodes])
        self.graph_token_shape = None
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_mapping,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        prefill_rows = self.rows - self.decode_rows
        context_lengths = tuple(kv - q for q, kv in zip(q_lens[num_decodes:], kv_lens[num_decodes:]))
        context_lengths = context_lengths if any(context_lengths) else ()
        context_offsets = tuple(slots[sum(q_lens[:index])] - (kv_lens[index] - q_lens[index])
            for index in range(num_decodes, len(kv_lens))) if context_lengths else ()
        self.prefill = DeepseekV2MLAFreshPrefillAttention(trace, model, pool, prefix + ":prefill",
            layer_index=layer_index, tp_size=tp_size, q_lens=q_lens[num_decodes:], cache=cache,
            slot_mapping=slot_mapping.narrow(0, self.decode_rows, prefill_rows), slots=slots[self.decode_rows:],
            rotary_cache=rotary_cache, reduction_group=reduction_group, reduction_config=reduction_config,
            context_lengths=context_lengths, context_offsets=context_offsets, context_chunk_rows=context_chunk_rows)
        self.decode = None
        if num_decodes:
            self.decode = DeepseekV2MLADecodeAttention(trace, model, pool, prefix + ":decode",
                layer_index=layer_index, tp_size=tp_size, kv_lens=kv_lens[:num_decodes], cache=cache,
                slot_mapping=slot_mapping.narrow(0, 0, self.decode_rows), slots=slots[:self.decode_rows],
                rotary_cache=rotary_cache, reduction_group=reduction_group, reduction_config=reduction_config,
                backend_source="vllm_0_30_flashmla_bf16", q_lens=q_lens[:num_decodes])

    def __call__(self, positions, hidden_states):
        physical = self.rows if self.graph_token_shape is None else self.graph_token_shape.model_input_tokens
        if hidden_states.shape != (physical, self.model.hidden_size) or positions.shape != (physical,):
            raise ValueError("mixed MLA inputs must cover decode and fresh prefill rows")
        with self.prefill.inputs(positions, hidden_states) as (query, latent, rope):
            if physical != self.rows:
                query, latent, rope = (value.narrow(0, 0, self.rows) for value in (query, latent, rope))
            self.update(latent, rope)
            output = self.trace.tensor(self.trace.unique_name(self.prefix + ":attention-output"),
                (physical, self.prefill.heads * self.model.v_head_dim), "bf16")
            count = self.rows - self.decode_rows
            self.prefill._forward_mha(query.narrow(0, self.decode_rows, count),
                latent.narrow(0, self.decode_rows, count), rope.narrow(0, self.decode_rows, count),
                output.narrow(0, self.decode_rows, count))
            if self.decode is not None:
                self.decode._decode_into(query.narrow(0, 0, self.decode_rows),
                    output.narrow(0, 0, self.decode_rows))
            return self.prefill.o_proj(output)[0]


def _v32_indexer_type(model, layer_index, mtp_indexer_type):
    if type(layer_index) is not int:
        raise ValueError("V32 layer requires an integer physical checkpoint index")
    if 0 <= layer_index < model.num_hidden_layers:
        if mtp_indexer_type is not None:
            raise ValueError("ordinary V32 layers cannot declare an MTP indexer branch")
        return model.indexer_types[layer_index]
    if (layer_index == model.num_hidden_layers and model.mtp_architecture == "glm_moe_dsa_mtp"
            and model.num_mtp_layers == 1 and mtp_indexer_type in ("full", "shared")):
        return mtp_indexer_type
    raise ValueError("V32 attention requires a target layer or its explicitly declared appended MTP indexer")


class DeepseekV32AttentionInput:
    """V32 attention.forward prefix before _sparse_indexer_and_attn.

    This is a source component, not a complete attention/model adapter. It
    keeps the source prefix frame live while the caller executes the rest
    of attention inside its context. Q absorption covers every model row
    before either prefill/decode backend is selected. CP/SP1, no HiSparse.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size,
                 cache, index_cache, topk_buffer, rotary_cache, index_rotary_cache, slot_mapping,
                 dummy_index_q, dummy_index_weights, mtp_indexer_type=None):
        if (model.model_type != "glm_moe_dsa" or model.weight_dtype not in ("bf16", "fp8")
                or model.compute_dtype != "bf16" or type(tp_size) is not int or tp_size not in (1, 8)):
            raise NotImplementedError("V32 attention prefix requires BF16-compute GLM TP1/TP8")
        indexer_type = _v32_indexer_type(model, layer_index, mtp_indexer_type)
        self.trace, self.model, self.prefix = trace, model, prefix
        self.local_heads = model.num_attention_heads // tp_size
        self.has_indexer = indexer_type == "full"
        module = f"model.layers.{layer_index}.self_attn."
        self.fused_qkv_a_proj = _source_linear(trace, model, pool, prefix + ":qkv-a",
            model.hidden_size, model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim,
            module + "fused_qkv_a_proj")
        self.wk_weights_proj, self.indexer_wq_b = None, None
        if self.has_indexer:
            self.wk_weights_proj = _source_linear(trace, model, pool, prefix + ":indexer-wk-weights",
                model.hidden_size, model.index_head_dim + model.index_n_heads, module + "indexer.wk_weights_proj")
            self.indexer_wq_b = _source_linear(trace, model, pool, prefix + ":indexer-wq-b",
                model.q_lora_rank, model.index_n_heads * model.index_head_dim, module + "indexer.wq_b")
        self.norm_rope = SymbolicDSANormRope(trace, prefix + ":norm-rope", pool, model,
            has_indexer=self.has_indexer, cache=cache, index_cache=index_cache, topk_buffer=topk_buffer,
            rotary_cache=rotary_cache, index_rotary_cache=index_rotary_cache, slot_mapping=slot_mapping)
        self.q_b_proj = _source_linear(trace, model, pool, prefix + ":q-b", model.q_lora_rank,
            self.local_heads * (model.qk_nope_head_dim + model.qk_rope_head_dim), module + "q_b_proj")
        self.q_absorb = SymbolicBatchedLinear(trace, prefix + ":q-absorb", pool,
            self.local_heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.fused_q = SymbolicDSAQuery(trace, prefix + ":fused-q", pool, model,
            has_indexer=self.has_indexer, rotary_cache=rotary_cache, index_rotary_cache=index_rotary_cache,
            dummy_index_q=dummy_index_q, dummy_index_weights=dummy_index_weights)

    @contextmanager
    def __call__(self, positions, hidden_states):
        model, trace = self.model, self.trace
        trace._check_tensor(positions)
        trace._check_tensor(hidden_states)
        if (len(hidden_states.shape) != 2 or hidden_states.dtype != "bf16"
                or hidden_states.shape[-1] != model.hidden_size
                or positions.shape != (hidden_states.shape[0],) or positions.dtype != "int64"
                or self.norm_rope.slot_mapping.shape != positions.shape
                or self.norm_rope.topk_buffer.shape[0] < hidden_states.shape[0]):
            raise ValueError("V32 attention prefix requires BF16 hidden rows and aligned INT64 positions")
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split((model.q_lora_rank, model.kv_lora_rank,
            model.qk_rope_head_dim), dim=-1)
        kw, index_k, index_weights = None, None, None
        if self.has_indexer:
            kw = self.wk_weights_proj(hidden_states)[0]
            index_k, index_weights = kw.split((model.index_head_dim, model.index_n_heads), dim=-1)
        rows = hidden_states.shape[0]
        output = trace.tensor(trace.unique_name(self.prefix + ":attention-output"),
            (rows, self.local_heads * model.v_head_dim), "bf16")
        kv_c_out = trace.tensor(trace.unique_name(self.prefix + ":kv-c-out"), kv_c.shape, "bf16")
        k_pe_out = trace.tensor(trace.unique_name(self.prefix + ":k-pe-out"), k_pe.shape, "bf16")
        q_c = self.norm_rope(positions, q_c, kv_c, k_pe, index_k, kv_c_out=kv_c_out, k_pe_out=k_pe_out)
        q = self.q_b_proj(q_c)[0].view_as_shape(rows, self.local_heads,
            model.qk_nope_head_dim + model.qk_rope_head_dim)
        q_nope, q_pe = q.split((model.qk_nope_head_dim, model.qk_rope_head_dim), dim=-1)
        ql_nope = self.q_absorb(q_nope.transpose(0, 1)).transpose(0, 1)
        index_q = None
        if self.has_indexer:
            index_q = self.indexer_wq_b(q_c)[0].view_as_shape(rows, model.index_n_heads, model.index_head_dim)
        index_q_fp8, index_weights_out, mqa_q = self.fused_q(positions,
            q_pe, ql_nope, index_q, index_weights)
        # The suspended source frame owns qkv_lora/kw/index_q automatically,
        # including locals not passed to the backend. No manual keep/free
        # table is needed when the caller enters the backend/output suffix.
        yield (q_c, q_nope, q_pe, ql_nope, index_q_fp8, index_weights_out, mqa_q,
            kv_c_out, k_pe_out, output)


class DeepseekV32Attention:
    """V32 BF16-compute latent attention, CP/SP1 and no HiSparse.

    A declared valid causal top-k domain is required. The backend's useful
    math/public output storage is represented, while its page remapping,
    ignored return and callee scratch stay explicit gaps. The module returns
    the un-reduced O projection, as the source decoder expects.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, q_lens, kv_lens, request_phases,
                 cache, index_cache, topk_buffer, rotary_cache, index_rotary_cache, slot_mapping,
                 dummy_index_q, dummy_index_weights, workspace, max_batched_tokens, selection_source,
                 prefill_attention_source=None, source_max_model_len=None, mtp_indexer_type=None,
                 num_speculative_tokens=0, reused_selection_kv_lens=None):
        indexer_type = _v32_indexer_type(model, layer_index, mtp_indexer_type)
        if mtp_indexer_type is not None and any(phase == "prefill" for phase in request_phases) and prefill_attention_source != "force_sparse_mqa":
            raise NotImplementedError("V32 MTP prefill requires explicit force_sparse_mqa; no fresh-dense inference")
        prefill_lengths = tuple((q, kv) for q, kv, phase in zip(q_lens, kv_lens, request_phases, strict=True)
            if phase == "prefill")
        if (prefill_attention_source is None and prefill_lengths
                and all(q == kv and kv <= model.index_topk for q, kv in prefill_lengths)):
            prefill_attention_source = "dense_mha_fresh"
        _dsa_source_phase(q_lens, kv_lens, request_phases, prefill_attention_source, num_speculative_tokens)
        self.dense_mha = prefill_attention_source == "dense_mha_fresh"
        if self.dense_mha and any(kv > model.index_topk for _, kv in prefill_lengths):
            raise NotImplementedError("fresh dense MHA requires every prefill context within top-k capacity")
        self.num_decodes = request_phases.count("decode")
        if not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace:
            raise ValueError("V32 attention requires this rank's explicitly initialized source workspace")
        if type(max_batched_tokens) is not int or max_batched_tokens < sum(q_lens):
            raise ValueError("V32 source batch capacity must cover all declared rows")
        if selection_source != "declared_causal_topk":
            raise NotImplementedError("V32 attention requires explicit valid top-k cardinality applicability")
        self.inputs = DeepseekV32AttentionInput(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=tp_size, cache=cache, index_cache=index_cache, topk_buffer=topk_buffer,
            rotary_cache=rotary_cache, index_rotary_cache=index_rotary_cache, slot_mapping=slot_mapping,
            dummy_index_q=dummy_index_q, dummy_index_weights=dummy_index_weights,
            mtp_indexer_type=mtp_indexer_type)
        self.model, self.heads = model, self.inputs.local_heads
        self.indexer = (SymbolicDSAIndexer(trace, prefix + ":indexer", pool, model,
            q_lens=q_lens, kv_lens=kv_lens, request_phases=request_phases, cache=index_cache,
            topk_buffer=topk_buffer, selection_source=selection_source, prefill_attention_source=prefill_attention_source,
            workspace=workspace if source_max_model_len is not None else None, source_max_model_len=source_max_model_len,
            num_speculative_tokens=num_speculative_tokens)
            if self.inputs.has_indexer else None)
        self.concat = SymbolicConcat(trace, prefix + ":concat-query", pool)
        self.q_concat_buffer, = workspace.get_simultaneous(
            ((max_batched_tokens, 64, model.kv_lora_rank + model.qk_rope_head_dim), "bf16"))
        self.sparse = None if self.dense_mha and not self.num_decodes else SymbolicDSASparseAttention(
            trace, prefix + ":sparse", pool, model, tp_size=tp_size,
            q_lens=q_lens[:self.num_decodes] if self.dense_mha else q_lens,
            kv_lens=kv_lens[:self.num_decodes] if self.dense_mha else kv_lens,
            request_phases=request_phases[:self.num_decodes] if self.dense_mha else request_phases, cache=cache,
            topk_buffer=topk_buffer, indexer_type=indexer_type,
            selection_source=selection_source, prefill_attention_source=None if self.dense_mha else prefill_attention_source,
            num_speculative_tokens=num_speculative_tokens, reused_selection_kv_lens=reused_selection_kv_lens)
        self.prefill_backend = (MLAFreshPrefillBackend(trace, model, pool, prefix,
            heads=self.heads, q_lens=q_lens[self.num_decodes:], cache=cache,
            module_path=f"model.layers.{layer_index}.self_attn")
            if self.dense_mha else None)
        self.decode_q_absorb = SymbolicBatchedLinear(trace, prefix + ":decode-q-absorb", pool,
            self.heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.v_up = SymbolicBatchedLinear(trace, prefix + ":v-up", pool, self.heads, model.kv_lora_rank, model.v_head_dim)
        self.o_proj = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.heads * model.v_head_dim, model.hidden_size, f"model.layers.{layer_index}.self_attn.o_proj")

    def _sparse_indexer_and_attn(self, index_q_fp8, index_weights_out, ql_nope, mqa_q, output):
        if self.indexer is not None:
            self.indexer(index_q_fp8, index_weights_out)
        rows = output.shape[0]
        # The source arena has 64 heads; concat_mla_q writes only the actual
        # head prefix, with the original padded token stride. Padding is not
        # initialized or charged as an extra copy by the mathematical model.
        q = self.q_concat_buffer.narrow(0, 0, rows).narrow(1, 0, self.heads)
        q = self.concat(ql_nope, mqa_q, out=q)
        attn_out = self.sparse(q)
        x = attn_out.view_as_shape(rows, self.heads, self.model.kv_lora_rank).transpose(0, 1)
        out = output.view_as_shape(rows, self.heads, self.model.v_head_dim).transpose(0, 1)
        self.v_up(x, out=out)

    def __call__(self, positions, hidden_states):
        with self.inputs(positions, hidden_states) as prepared:
            q_c, q_nope, q_pe, ql_nope, index_q_fp8, index_weights_out, mqa_q, kv_c, k_pe, output = prepared
            if self.dense_mha:
                if self.indexer is not None:
                    self.indexer(index_q_fp8, index_weights_out)
                # V32 calls forward_impl with a fresh full-head Q concat.
                # The earlier all-T Q absorption/fused_q remain source work.
                mha_q = self.concat(q_nope, mqa_q)
                self._forward_dense_mha(mha_q, kv_c, k_pe, output)
            else:
                self._sparse_indexer_and_attn(index_q_fp8, index_weights_out, ql_nope, mqa_q, output)
            return self.o_proj(output)[0]

    def _forward_dense_mha(self, query, latent, rope, output):
        """MLACommon.forward_impl executes prefill first, then its decode prefix."""
        trace, model, decodes = self.inputs.trace, self.model, self.num_decodes
        rows = query.shape[0] - decodes
        self.prefill_backend(query.narrow(0, decodes, rows), latent.narrow(0, decodes, rows),
            rope.narrow(0, decodes, rows).unsqueeze(1), output.narrow(0, decodes, rows))
        if decodes:
            decode_query = query.narrow(0, 0, decodes)
            q_nope, q_pe = decode_query.split((model.qk_nope_head_dim, model.qk_rope_head_dim))
            # The common wrapper absorbs these rows again; the V32 prefix's
            # earlier all-T absorption is still live source work, not its input.
            absorbed = trace.tensor(trace.unique_name(self.decode_q_absorb.name + ":output"),
                (decodes, self.heads, model.kv_lora_rank), "bf16")
            self.decode_q_absorb(q_nope.transpose(0, 1), out=absorbed.transpose(0, 1))
            packed = self.q_concat_buffer.narrow(0, 0, decodes).narrow(1, 0, self.heads)
            packed = self.concat(absorbed, q_pe, out=packed)
            attn_out = self.sparse(packed)
            self.v_up(attn_out.transpose(0, 1), out=output.narrow(0, 0, decodes).view_as_shape(
                decodes, self.heads, model.v_head_dim).transpose(0, 1))


class DeepseekV32DecoderLayer:
    """PP/SP1 source layer with explicit all-reduce → residual RMSNorm.

    This declares the ordinary branch of fused_allreduce_rms_norm, rather
    than claiming selection of its optional FlashInfer fusion/provider.
    """

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index, self_attn, resolution, invocation_id,
                 expert_parallel_size=1, input_moe_invocation_id=None):
        self.group, self.config, self.pool, self.prefix = group, config, pool, prefix
        self.self_attn = self_attn
        self.input_layernorm = SymbolicRMSNorm(trace, prefix + ":input-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        self.post_attention_layernorm = SymbolicRMSNorm(trace, prefix + ":post-attention-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        self.input_reduce_norm = (self.input_layernorm if group is None else
            SymbolicAllReduceRMSNorm(self.input_layernorm, group, config,
                name=prefix + ":input-all-reduce", pool_name=pool, moe_invocation_id=input_moe_invocation_id))
        self.post_attention_reduce_norm = (self.post_attention_layernorm if group is None else
            SymbolicAllReduceRMSNorm(self.post_attention_layernorm, group, config,
                name=prefix + ":attention-all-reduce", pool_name=pool))
        tp_size = 1 if group is None else len(group.traces)
        uses_moe = (model.layer_uses_moe(layer_index) if layer_index < model.num_hidden_layers
            else model.mtp_layer_uses_moe(layer_index - model.num_hidden_layers))
        if uses_moe:
            self.mlp = DeepseekV32MoE(trace, model, pool, prefix + ":mlp", tp_size=tp_size,
                layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
                expert_parallel_size=expert_parallel_size,
                expert_rank=group.participants.index(trace.rank) if expert_parallel_size > 1 else 0)
        else:
            self.mlp = DeepseekV2MLP(trace, model, pool, prefix + ":mlp", tp_size=tp_size,
                intermediate_size=model.intermediate_size, module_path=f"model.layers.{layer_index}.mlp", reduce_results=False)

    def __call__(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_reduce_norm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_reduce_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV32MTPPredictorLayer:
    """The appended V32 draft layer with explicit arenas and source branches.

    The caller supplies each pass's draft IDs, routing and full/shared indexer
    declaration. This component does not prepare proposer inputs or decide
    acceptance, sample tokens, bind a provider or assert normal capacity.
    """

    def __init__(self, trace, model, pool, prefix, *, group, collective, gather,
                 input_token_ids, q_lens, kv_lens, request_phases, slots, cache_capacity_tokens,
                 resolution, invocation_id, source_profile, source_max_model_len, max_batched_tokens,
                 workspace, topk_buffer, cache, index_cache, rotary_cache, index_rotary_cache,
                 slot_tensor, indexer_type, prefill_attention_source, selection_source="declared_causal_topk",
                 input_mode="token_ids", input_norm=None, num_speculative_tokens=0,
                 reused_selection_kv_lens=None, startup_attention_factory=None):
        from .traced_operators import SymbolicV32EHNorm, SymbolicV32EmbedEHNorm
        if (model.model_type != "glm_moe_dsa" or model.mtp_architecture != "glm_moe_dsa_mtp"
                or model.num_mtp_layers != 1 or model.num_hidden_layers != 78
                or model.compute_dtype != "bf16" or model.weight_dtype != _glm_source_weight_dtype(source_profile)
                or not model.mtp_layer_uses_moe(0)):
            raise NotImplementedError("V32 draft requires its accurate BF16-compute appended single MoE layer and weight source")
        if group is None:
            if collective is not None or gather is not None:
                raise ValueError("TP1 V32 draft cannot declare collective configurations")
            tp_size = 1
        elif isinstance(group, ForwardCollectiveGroup) and len(group.traces) == 8 and any(trace is item for item in group.traces):
            tp_size = 8
        else:
            raise NotImplementedError("V32 draft requires TP1 or this rank's explicit TP8 group")
        if (not isinstance(q_lens, tuple) or not q_lens or any(type(q) is not int or q < 1 for q in q_lens)
                or not isinstance(kv_lens, tuple) or len(kv_lens) != len(q_lens)
                or any(type(k) is not int or not q <= k <= model.max_position_embeddings
                    for q, k in zip(q_lens, kv_lens, strict=True))
                or not isinstance(request_phases, tuple) or len(request_phases) != len(q_lens)
                or any(phase not in ("decode", "prefill")
                    for phase, q in zip(request_phases, q_lens, strict=True))):
            raise ValueError("V32 draft requires complete positive query/context lengths and explicit backend phases")
        if startup_attention_factory is not None:
            from .traced_glm_mtp import GLMStartupAttentionFactory
            if type(startup_attention_factory) is not GLMStartupAttentionFactory:
                raise TypeError("V32 startup requires its typed no-metadata attention factory")
            startup_attention_factory.validate(rows=sum(q_lens), max_batched_tokens=max_batched_tokens,
                source_max_model_len=source_max_model_len, cache_capacity_tokens=cache_capacity_tokens,
                slot_mapping=slots, selection_source=selection_source, prefill_attention_source=prefill_attention_source)
            if (any(value is not None for value in (cache, index_cache, slot_tensor, reused_selection_kv_lens))
                    or indexer_type != "full" or input_mode != "token_ids"):
                raise ValueError("V32 startup draft has no request caches/slots or reused selection")
        if (startup_attention_factory is None and any(phase == "prefill" for phase in request_phases)
                and prefill_attention_source != "force_sparse_mqa"):
            raise NotImplementedError("V32 multiquery/prefill draft requires explicit force_sparse_mqa metadata")
        if startup_attention_factory is None:
            _dsa_source_phase(q_lens, kv_lens, request_phases, prefill_attention_source, num_speculative_tokens)
        rows = sum(q_lens)
        if startup_attention_factory is None and (type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
                or type(max_batched_tokens) is not int or max_batched_tokens < rows
                or type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
                or not isinstance(slots, tuple) or len(slots) != rows or len(set(slots)) != rows
                or any(type(slot) is not int or not 0 <= slot < cache_capacity_tokens for slot in slots)):
            raise ValueError("V32 draft requires explicit source limits and distinct logical cache slots")
        if (not isinstance(input_token_ids, tuple) or len(input_token_ids) != rows
                or any(type(token) is not int or not 0 <= token < model.vocab_size for token in input_token_ids)
                or indexer_type not in ("full", "shared") or input_mode not in ("token_ids", "inputs_embeds")):
            raise ValueError("V32 draft requires its own valid IDs, full/shared indexer and explicit input mode")
        if not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace:
            raise ValueError("V32 draft workspace must belong to the current rank")
        cache_arenas = (() if startup_attention_factory is not None else (
            (cache, (cache_capacity_tokens, model.kv_lora_rank + model.qk_rope_head_dim), "bf16"),
            (index_cache, (cache_capacity_tokens, 132), "uint8"), (slot_tensor, (rows,), "int64")))
        for tensor, shape, dtype in (*cache_arenas,
                (topk_buffer, (max_batched_tokens, model.index_topk), "int32"),
                (rotary_cache, (model.max_position_embeddings, model.qk_rope_head_dim), "bf16"),
                (index_rotary_cache, (model.max_position_embeddings, model.qk_rope_head_dim), "bf16")):
            trace._check_tensor(tensor)
            if tensor.shape != shape or tensor.dtype != dtype:
                raise ValueError("V32 draft requires exact typed cache/top-k/rotary/slot arenas")
        self.trace, self.model, self.rows, self.input_mode = trace, model, rows, input_mode
        self.position_values = ((0,) * rows if startup_attention_factory is not None else
            tuple(p for q, k in zip(q_lens, kv_lens, strict=True) for p in range(k - q, k)))
        epsilon = model.target_semantics.rms_norm_eps
        self.embed_tokens, self.eh_norm, self.embed_eh_norm = None, None, None
        norm_type = SymbolicV32EmbedEHNorm if input_mode == "token_ids" and group is None else SymbolicV32EHNorm
        if input_norm is not None and (type(input_norm) is not norm_type or input_norm.trace is not trace
                or input_norm.width != model.hidden_size or input_norm.epsilon != epsilon
                or (norm_type is SymbolicV32EmbedEHNorm and input_norm.vocab != model.vocab_size)):
            raise ValueError("V32 recurrent draft input norm must share this rank's exact typed weight metadata")
        if input_mode == "token_ids" and group is None:
            self.embed_eh_norm = (input_norm if input_norm is not None else
                SymbolicV32EmbedEHNorm(trace, prefix + ":embed-eh-norm", pool, model.hidden_size, model.vocab_size, epsilon))
        else:
            if input_mode == "token_ids":
                self.embed_tokens = SymbolicVocabParallelEmbedding(trace, prefix + ":embedding", pool,
                    model.hidden_size, model.vocab_size, group, collective, token_ids=input_token_ids)
            self.eh_norm = (input_norm if input_norm is not None else
                SymbolicV32EHNorm(trace, prefix + ":eh-norm", pool, model.hidden_size, epsilon))
        self.input_norm = self.embed_eh_norm if self.embed_eh_norm is not None else self.eh_norm
        self.input_token_ids = input_token_ids
        self.eh_proj = SymbolicLinear(trace, prefix + ":eh-proj", pool, 2 * model.hidden_size, model.hidden_size)
        dummy_q = trace.tensor(prefix + ":dummy-index-q", (1, 1, 1), "bf16", persistent=True)
        dummy_weights = trace.tensor(prefix + ":dummy-index-weights", (1, 1), "fp32", persistent=True)
        attention = (startup_attention_factory.attention(trace, model, pool, prefix + ":attention",
            layer_index=model.num_hidden_layers, tp_size=tp_size, workspace=workspace,
            rotary_cache=rotary_cache, topk_buffer=topk_buffer) if startup_attention_factory is not None else
            DeepseekV32Attention(trace, model, pool, prefix + ":attention",
            layer_index=model.num_hidden_layers, tp_size=tp_size, q_lens=q_lens, kv_lens=kv_lens,
            request_phases=request_phases, cache=cache, index_cache=index_cache if indexer_type == "full" else None,
            topk_buffer=topk_buffer, rotary_cache=rotary_cache, index_rotary_cache=index_rotary_cache,
            slot_mapping=slot_tensor, dummy_index_q=dummy_q, dummy_index_weights=dummy_weights,
            workspace=workspace, max_batched_tokens=max_batched_tokens, selection_source=selection_source,
            prefill_attention_source=prefill_attention_source, source_max_model_len=source_max_model_len,
            mtp_indexer_type=indexer_type, num_speculative_tokens=num_speculative_tokens,
            reused_selection_kv_lens=reused_selection_kv_lens))
        self.mtp_block = DeepseekV32DecoderLayer(trace, model, pool, prefix + ":decoder", group, collective,
            layer_index=model.num_hidden_layers, self_attn=attention, resolution=resolution, invocation_id=invocation_id)
        norm = SymbolicRMSNorm(trace, prefix + ":final-norm", pool, model.hidden_size, epsilon)
        self.final_norm = (norm if group is None else SymbolicAllReduceRMSNorm(norm, group, collective,
            name=prefix + ":final-all-reduce", pool_name=pool))
        self.head = (SymbolicLinear(trace, prefix + ":head", pool, model.hidden_size, model.vocab_size,
            kind=WorkloadKind.LM_HEAD) if group is None else SymbolicTPLogitsProcessor(trace, prefix + ":head", pool,
                hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather))

    def __call__(self, ids, positions, previous_hidden_states, *, position_values, inputs_embeds=None):
        for tensor in (ids, positions, previous_hidden_states):
            self.trace._check_tensor(tensor)
        if (ids.shape != (self.rows,) or ids.dtype != "int32"
                or positions.shape != (self.rows,) or positions.dtype != "int64"
                or previous_hidden_states.shape != (self.rows, self.model.hidden_size) or previous_hidden_states.dtype != "bf16"
                or not isinstance(position_values, tuple) or any(type(value) is not int for value in position_values)
                or position_values != self.position_values):
            raise ValueError("V32 draft inputs must match its own typed rows and exact declared context positions")
        if self.input_mode == "token_ids":
            if inputs_embeds is not None:
                raise ValueError("V32 token-ID draft cannot silently switch to external embeddings")
            if self.embed_eh_norm is not None:
                eh = self.embed_eh_norm(ids, positions, previous_hidden_states,
                    position_values=position_values, input_token_ids=self.input_token_ids)
            else:
                eh = self.eh_norm(positions, self.embed_tokens(ids), previous_hidden_states, position_values=position_values)
        else:
            if inputs_embeds is None:
                raise ValueError("V32 inputs_embeds draft requires its explicit tensor port")
            eh = self.eh_norm(positions, inputs_embeds, previous_hidden_states, position_values=position_values)
        hidden_states = self.eh_proj(eh)[0]
        hidden_states, residual = self.mtp_block(positions, hidden_states, None)
        hidden_states, _ = self.final_norm(hidden_states, residual)
        return hidden_states, hidden_states

    def compute_logits(self, hidden_states):
        # V32 mtp.py consumes the post-final-norm hidden without another norm.
        return self.head(hidden_states)[0] if isinstance(self.head, SymbolicLinear) else self.head(hidden_states)


def _glm_source_weight_dtype(source_profile):
    if source_profile == "vllm_0_30_v32_bf16_sync_explicit_allreduce":
        return "bf16"
    if source_profile == "vllm_0_30_v32_fp8_sync_explicit_allreduce":
        return "fp8"
    raise NotImplementedError("GLM decoder requires an explicit modular BF16/FP8 all-reduce source profile")


class DeepseekV32Model:
    """Pure-text GLM decoder or contiguous PP stage, without aux/MTP."""

    def __init__(self, trace, model, pool, group, config, *, token_ids, q_lens, kv_lens, request_phases,
                 slot_mapping, cache_capacity_tokens, source_max_model_len, max_batched_tokens,
                 resolutions, invocation_ids, workspace_initial_capacity_bytes, selection_source, source_profile,
                 prefill_attention_source, model_input_order, num_speculative_tokens=0, startup_attention_factory=None,
                 expert_parallel_size=1, layer_range=None):
        weight_dtype = _glm_source_weight_dtype(source_profile)
        if (model.model_type != "glm_moe_dsa" or model.weight_dtype != weight_dtype or model.compute_dtype != "bf16"
                or model.target_semantics.rope_layout != "interleaved"
                or model.target_semantics.indexer_rope_layout != "interleaved"):
            raise NotImplementedError("GLM decoder requires checkpoint/source weight dtype agreement and matching interleaved RoPE")
        self.group, self.config, self.pool = group, config, pool
        start, end = (0, model.num_hidden_layers) if layer_range is None else layer_range
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= model.num_hidden_layers):
            raise ValueError("GLM stage requires a nonempty contiguous checkpoint layer range")
        self.first_stage, self.last_stage = start == 0, end == model.num_hidden_layers
        if not self.first_stage and model.indexer_types[start] != "full":
            raise NotImplementedError("each GLM stage must start with its source full indexer")
        if startup_attention_factory is not None:
            if layer_range is not None:
                raise NotImplementedError("GLM startup factory does not bind pipeline stages")
            from .traced_glm_mtp import GLMStartupAttentionFactory
            if type(startup_attention_factory) is not GLMStartupAttentionFactory:
                raise TypeError("GLM startup requires its typed no-metadata attention factory")
            startup_attention_factory.validate(rows=sum(q_lens), max_batched_tokens=max_batched_tokens,
                source_max_model_len=source_max_model_len, cache_capacity_tokens=cache_capacity_tokens,
                slot_mapping=slot_mapping, selection_source=selection_source, prefill_attention_source=prefill_attention_source)
        self.embed_tokens = (SymbolicEmbedding(trace, "embedding", pool, model.hidden_size, "bf16",
            model_input_order=model_input_order) if group is None else
            SymbolicVocabParallelEmbedding(trace, "embedding", pool, model.hidden_size,
                model.vocab_size, group, config, token_ids=token_ids,
                model_input_order=model_input_order if trace.rank == group.participants[0] else None)) if self.first_stage else None
        self.workspace = ForwardWorkspace(trace, "worker.workspace", alignment_bytes=256,
            initial_capacity_bytes=workspace_initial_capacity_bytes, locked=False)
        topk = trace.tensor("topk", (max_batched_tokens, model.index_topk), "int32", persistent=True)
        rope = trace.tensor("rotary_cache", (model.max_position_embeddings, model.qk_rope_head_dim), "bf16", persistent=True)
        # Both source get_rope calls have equal arguments in this declared
        # GLM branch, and the factory cache returns the same object.
        index_rope = rope
        slots = (trace.tensor("slot_mapping", (sum(q_lens),), "int64", persistent=True)
            if startup_attention_factory is None else None)
        self.slot_mapping = slots
        dummy_q = trace.tensor("dummy:index-q", (1, 1, 1), "bf16", persistent=True)
        dummy_weights = trace.tensor("dummy:index-weights", (1, 1), "fp32", persistent=True)
        self.layers = []
        for layer in range(start, end):
            prefix = f"layer-{layer}"
            cache = (trace.tensor(prefix + ":cache", (cache_capacity_tokens,
                model.kv_lora_rank + model.qk_rope_head_dim), "bf16", persistent=True)
                if startup_attention_factory is None else None)
            index_cache = (trace.tensor(prefix + ":index-cache", (cache_capacity_tokens, 132), "uint8", persistent=True)
                if startup_attention_factory is None and model.indexer_types[layer] == "full" else None)
            attention = (startup_attention_factory.attention(trace, model, pool, prefix + ":attention",
                layer_index=layer, tp_size=1 if group is None else len(group.traces), workspace=self.workspace,
                rotary_cache=rope, topk_buffer=topk) if startup_attention_factory is not None else
                DeepseekV32Attention(trace, model, pool, prefix + ":attention", layer_index=layer,
                tp_size=1 if group is None else len(group.traces), q_lens=q_lens, kv_lens=kv_lens, request_phases=request_phases,
                cache=cache, index_cache=index_cache, topk_buffer=topk, rotary_cache=rope, index_rotary_cache=index_rope,
                slot_mapping=slots, dummy_index_q=dummy_q, dummy_index_weights=dummy_weights, workspace=self.workspace,
                max_batched_tokens=max_batched_tokens, selection_source=selection_source,
                prefill_attention_source=prefill_attention_source, source_max_model_len=source_max_model_len,
                num_speculative_tokens=num_speculative_tokens))
            self.layers.append(DeepseekV32DecoderLayer(trace, model, pool, prefix, group, config, layer_index=layer,
                self_attn=attention, resolution=resolutions[layer], invocation_id=invocation_ids[layer],
                expert_parallel_size=expert_parallel_size, input_moe_invocation_id=(invocation_ids[layer - 1]
                    if expert_parallel_size > 1 and layer > 0 and model.layer_uses_moe(layer - 1) else None)))
        self.norm = (SymbolicRMSNorm(trace, "final-norm", pool, model.hidden_size, model.target_semantics.rms_norm_eps)
            if self.last_stage else None)
        self.rotary_cache, self.index_rotary_cache = rope, index_rope
        self.final_reduce_norm = (self.norm if group is None or not self.last_stage else
            SymbolicAllReduceRMSNorm(self.norm, group, config, name="final-all-reduce", pool_name=pool,
                moe_invocation_id=invocation_ids[-1] if expert_parallel_size > 1 else None))

    def __call__(self, input_ids, positions, *, intermediate_tensors=None):
        if self.first_stage:
            if intermediate_tensors is not None:
                raise ValueError("first GLM stage cannot consume pipeline intermediates")
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            if not isinstance(intermediate_tensors, dict) or set(intermediate_tensors) != {"hidden_states", "residual"}:
                raise ValueError("non-first GLM stage requires both explicit intermediate ports")
            hidden_states, residual = intermediate_tensors["hidden_states"], intermediate_tensors["residual"]
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not self.last_stage:
            return {"hidden_states": hidden_states, "residual": residual}
        hidden_states, _ = self.final_reduce_norm(hidden_states, residual)
        return hidden_states


def _source_expert_resolutions(model, step, pool, expert_routing, invocation_ids, rows):
    """Common source routing identity; no model-specific histograms or IDs."""
    from src.serving.schedule.schedule_step import schedule_step_sha256
    from .runtime_inputs import ExpertRoutingInput
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("virtual MoE source requires canonical explicit routing input")
    digest = schedule_step_sha256(step)
    config = pool.parallel_plan.config
    active = tuple(layer for layer in range(model.num_hidden_layers) if model.layer_uses_moe(layer))
    if not active:
        return (None,) * model.num_hidden_layers
    layers = tuple((layer, invocation_ids[layer], ordinal,
        (pool.physical_ranks_for_pipeline_stage(pool.parallel_plan.layer_to_pipeline_stage[layer])
            if config.enable_expert_parallel else
            (pool.physical_ranks_for_pipeline_stage(pool.parallel_plan.layer_to_pipeline_stage[layer])[0],)))
        for ordinal, layer in enumerate(active))
    resolved = expert_routing.resolve_layers(layers, step_id=step.step_id,
        schedule_step_sha256=digest, tokens=rows, num_experts=model.num_experts,
        topk=model.num_experts_per_tok, model_config_sha256=model.config_sha256,
        pool_name=pool.name, tensor_parallel_size=config.tensor_parallel_size,
        expert_parallel_size=config.expert_parallel_size, expert_placement_strategy="linear",
        assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
            if expert_routing.source_kind == "explicit_assumption" else None))
    by_layer = dict(zip(active, resolved, strict=True))
    return tuple(by_layer[layer] if layer in by_layer else None for layer in range(model.num_hidden_layers))



def _plain_greedy_target(trace, model, pool, group, gather, hidden_states, indices, q_lens,
                         *, model_execution_version="vllm_0_30_0", sample_logits=True, cast_logits=False):
    """Shared typed target boundary, keeping indices in the caller's frame."""
    if type(sample_logits) is not bool or type(cast_logits) is not bool or sample_logits and cast_logits:
        raise TypeError("target output flags must be exclusive booleans")
    cumulative, selected = 0, []
    for count in q_lens:
        cumulative += count
        selected.append(cumulative - 1)
    sample_hidden_states = SymbolicLogitsRowSelect(trace, "logits-row-select", pool)(
        hidden_states, indices, selected_indices=tuple(selected), model_execution_version=model_execution_version)
    if group is None:
        logits, _ = SymbolicLinear(trace, "lm-head", pool, model.hidden_size,
            model.vocab_size, kind=WorkloadKind.LM_HEAD)(sample_hidden_states)
    else:
        processor = SymbolicTPLogitsProcessor(trace, "lm-head", pool, hidden_size=model.hidden_size,
            vocab_size=model.vocab_size, group=group, config=gather, model_execution_version=model_execution_version)
        logits = processor(sample_hidden_states)
    return (SymbolicGreedy(trace, "target-sampler", pool)(logits, model_execution_version=model_execution_version,
        sample_logits=sample_logits) if sample_logits or cast_logits else logits)


def _trace_glm_dsa_source(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
                        cache_capacity_tokens, source_max_model_len, max_batched_tokens,
                        workspace_initial_capacity_bytes, selection_source, source_profile, target_sampling_source,
                        prefill_attention_source=None, source_prompt_lengths=None, include_lm_head=False):
    """One GLM source program, optionally including its plain target chain."""
    from .runtime_inputs import ExpertRoutingInput
    from .runtime_inputs import vllm_0_30_source_model_inputs
    weight_dtype = _glm_source_weight_dtype(source_profile)
    if type(include_lm_head) is not bool:
        raise TypeError("GLM include_lm_head must be boolean")
    if target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16"):
        raise NotImplementedError("symbolic GLM target supports only its declared BF16 plain greedy chain")
    config = pool.parallel_plan.config
    if (model.model_type != "glm_moe_dsa" or model.compute_dtype != "bf16" or model.weight_dtype != weight_dtype
            or pool.hardware_id != "nvidia_h200" or config.tensor_parallel_size not in (1, 8)
            or len(pool.ranks) != config.tensor_parallel_size
            or (config.enable_expert_parallel and config.expert_parallel_size != 8)
            or any(value != 1 for value in (config.pipeline_parallel_size,
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("symbolic GLM decoder requires H200 checkpoint/source agreement, TP1/TP8 PP/SP/CP/DP1, EP-off or EP8")
    if (not step.requests or step.pool_name != pool.name
            or any(r.phase not in ("decode", "prefill") or r.scheduled_tokens <= 0 or r.mtp_proposed_tokens
                or r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests)):
        raise NotImplementedError("symbolic GLM decoder requires all-decode single-token requests with MTP off")
    fresh_dense = (prefill_attention_source is None and any(r.phase == "prefill" for r in step.requests)
        and all((r.phase == "prefill" and r.computed_tokens == 0 and 0 < r.scheduled_tokens <= model.index_topk)
            or (r.phase == "decode" and r.scheduled_tokens == 1) for r in step.requests))
    if any(r.scheduled_tokens > 1 for r in step.requests) and prefill_attention_source != "force_sparse_mqa" and not fresh_dense:
        raise NotImplementedError("GLM all-decode source needs explicit force_sparse_mqa for prefill/mixed")
    rows = sum(r.scheduled_tokens for r in step.requests)
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(value) is not int for value in values):
            raise ValueError(f"symbolic GLM {name} requires one explicit integer per model row")
    if any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("GLM decoder token IDs must belong to the accurate vocabulary")
    if positions != tuple(p for r in step.requests for p in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens)):
        raise ValueError("GLM decoder positions must match request contexts")
    step, input_ids, model_input_order = vllm_0_30_source_model_inputs(step,
        embedding_token_ids=input_ids, prompt_lengths=source_prompt_lengths)
    positions = tuple(positions[row] for row in model_input_order["token_row_permutation"])
    slot_mapping = tuple(slot_mapping[row] for row in model_input_order["token_row_permutation"])
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    phases = tuple(model_input_order["model_attention_phases"])
    if fresh_dense and "prefill" in phases:
        prefill_attention_source = "dense_mha_fresh"
    _dsa_source_phase(q_lens, kv_lens, phases, prefill_attention_source)
    if (type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
            or type(max_batched_tokens) is not int or max_batched_tokens < rows):
        raise ValueError("GLM source context and batch limits must contain every declared row")
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slot_mapping)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slot_mapping)):
        raise ValueError("GLM logical cache must cover contexts and distinct valid write slots")
    if (not isinstance(workspace_initial_capacity_bytes, tuple) or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("GLM source workspace requires explicit entry capacities for every physical rank")
    if selection_source != "declared_causal_topk":
        raise NotImplementedError("GLM decoder requires declared valid causal top-k cardinality, not inferred IDs")
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("GLM decoder requires canonical explicit routing input")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = None if len(traces) == 1 else ForwardCollectiveGroup(traces, namespace=step.step_id)
    collective = None if group is None else pool.collective_for("all_reduce")
    target_head = include_lm_head or target_sampling_source is not None
    gather = None if not target_head or group is None else pool.collective_for("all_gather")
    invocation_ids = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocation_ids, rows)

    def rank_forward(trace, capacity):
        network = DeepseekV32Model(trace, model, pool.name, group, collective, token_ids=input_ids,
            q_lens=q_lens, kv_lens=kv_lens, request_phases=phases,
            slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            resolutions=resolutions, invocation_ids=invocation_ids, workspace_initial_capacity_bytes=capacity,
            selection_source=selection_source, source_profile=source_profile,
            prefill_attention_source=prefill_attention_source, model_input_order=model_input_order,
            expert_parallel_size=config.expert_parallel_size)
        ids = trace.tensor("input_ids", (rows,), "int32")
        pos = trace.tensor("positions", (rows,), "int64")
        indices = trace.tensor("logits_indices", (len(q_lens),), "int32") if target_head else None
        hidden_states = network(ids, pos)
        if target_head:
            # _prepare_inputs supplies INT32 query_start_loc[1:]-1. Even
            # identity indices materialize sample_hidden_states in execute_model.
            sampled = _plain_greedy_target(trace, model, pool.name, group, gather, hidden_states, indices, q_lens,
                sample_logits=target_sampling_source == "vllm_0_30_plain_greedy_bf16",
                cast_logits=target_sampling_source == "vllm_0_30_plain_sampler_bf16")

    for trace, capacity in zip(traces, workspace_initial_capacity_bytes, strict=True):
        rank_forward(trace, capacity)
    return traces[0] if group is None else group


def trace_glm_dsa_decoder(model, step, pool, expert_routing, **source_inputs):
    """GLM model.forward scope only; no logits/sampler or Serving dispatch."""
    return _trace_glm_dsa_source(model, step, pool, expert_routing, target_sampling_source=None, **source_inputs)


def trace_glm_dsa_forward(model, step, pool, expert_routing, *, target_sampling_source, **source_inputs):
    """GLM decoder → selected rows → BF16 TP head → plain INT32 greedy.

    Decode or an explicitly forced sparse-MQA prefill/mixed source domain.
    Metadata producer/library/compiler services, full worker memory and
    actual provider remain separate; routing pricing refusal is unchanged.
    """
    if target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16"):
        raise NotImplementedError("complete GLM target requires an explicit plain BF16 greedy source")
    return _trace_glm_dsa_source(model, step, pool, expert_routing,
        target_sampling_source=target_sampling_source, include_lm_head=True, **source_inputs)


class DeepseekV4MLP:
    """DeepseekV4MLP.forward with EP-off TP shards and reduce_results=False."""

    def __init__(self, trace, model, pool, prefix, *, tp_size, linear_source_profile):
        if model.model_type != "deepseek_v4" or type(tp_size) is not int or tp_size not in (4, 8):
            raise NotImplementedError("symbolic V4 shared MLP currently requires EP-off TP4/8")
        intermediate = model.moe_intermediate_size * model.num_shared_experts
        # n_local = ceil((intermediate // block_size) / TP), as in source.
        if intermediate % 128:
            raise NotImplementedError("V4 shared block-FP8 source requires complete checkpoint128 blocks")
        local = -(-(intermediate // 128) // tp_size) * 128
        self.gate_up_proj = SymbolicBlockFP8Linear(trace, prefix + ":gate-up", pool,
            model.hidden_size, 2 * local, source_profile=linear_source_profile)
        self.act_fn = SymbolicSiluAndMul(trace, prefix + ":silu-mul", pool, swiglu_limit=model.swiglu_limit)
        self.down_proj = SymbolicBlockFP8Linear(trace, prefix + ":down", pool,
            local, model.hidden_size, source_profile=linear_source_profile)

    def __call__(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV4AttentionInput:
    """V4 attention input projections and fused Q/KV norm, all ratios.

    Adapted from attention.py _run_parallel_input_projections and
    _split_qkv_and_norm. The explicit synchronous source profile preserves
    value dependencies and frame references; it does not model aux-stream
    retirement, CUDA capture or an installed linear provider.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, linear_source_profile):
        if (model.model_type != "deepseek_v4" or type(layer_index) is not int
                or not 0 <= layer_index < model.num_hidden_layers):
            raise ValueError("V4 attention input requires a checkpoint layer")
        self.compress_ratio = max(1, model.compress_ratios[layer_index])
        self.fused_wqa_wkv = SymbolicBlockFP8Linear(trace, prefix + ":fused-wqa-wkv", pool,
            model.hidden_size, model.q_lora_rank + model.head_dim, source_profile=linear_source_profile)
        self.compressor_projection = None
        self.indexer_weights_proj = None
        self.indexer_compressor_projection = None
        if self.compress_ratio > 1:
            coefficients = 2 if self.compress_ratio == 4 else 1
            self.compressor_projection = SymbolicLinear(trace, prefix + ":compressor-kv-score", pool,
                model.hidden_size, 2 * coefficients * model.head_dim, output_dtype="fp32")
        if self.compress_ratio == 4:
            self.indexer_weights_proj = SymbolicLinear(trace, prefix + ":indexer-weights", pool,
                model.hidden_size, model.index_n_heads)
            self.indexer_compressor_projection = SymbolicLinear(trace, prefix + ":indexer-kv-score", pool,
                model.hidden_size, 4 * model.index_head_dim, output_dtype="fp32")
        self.norm = SymbolicV4QKVNorm(trace, prefix + ":q-kv-rmsnorm", pool, model)

    def _run_parallel_input_projections(self, hidden_states):
        # Source execute_in_parallel returns every auxiliary output, including
        # C4 indexer projections later unused by its short-context return.
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        kv_score = None if self.compressor_projection is None else self.compressor_projection(hidden_states)[0]
        indexer_weights = None if self.indexer_weights_proj is None else self.indexer_weights_proj(hidden_states)[0]
        indexer_kv_score = (None if self.indexer_compressor_projection is None
            else self.indexer_compressor_projection(hidden_states)[0])
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def __call__(self, hidden_states):
        qr_kv, kv_score, indexer_kv_score, indexer_weights = self._run_parallel_input_projections(hidden_states)
        qr, qr_scale, kv = self.norm(qr_kv)
        if qr_scale is not None:
            raise NotImplementedError("V4 SM90 source does not prequantize the q-lora norm result")
        return qr, kv, kv_score, indexer_kv_score, indexer_weights


class DeepseekV4MoERunner:
    """Synchronous modular MoERunner source frames, no DP/EP/LoRA/overlap.

    Separate Python frames preserve the source release of logits and route
    outputs before the final shared-add/all-reduce. Unknown expert workspace
    remains declared on its node; this is not complete memory acceptance.
    """

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index,
                 resolution, invocation_id, linear_source_profile, workspace=None, source_multiprocessors=None):
        self.gate = SymbolicLinear(trace, prefix + ":router", pool,
            model.hidden_size, model.num_experts, kind=WorkloadKind.ROUTER, output_dtype="fp32")
        self.shared_experts = DeepseekV4MLP(trace, model, pool, prefix + ":shared", tp_size=len(group.traces),
            linear_source_profile=linear_source_profile)
        self.router = SymbolicV4Routing(trace, prefix + ":route", pool, model, layer_index=layer_index)
        self.routed_experts = SymbolicV4RoutedExperts(trace, prefix + ":routed", pool, model,
            tp_size=len(group.traces), resolution=resolution, invocation_id=invocation_id, workspace=workspace,
            source_multiprocessors=source_multiprocessors)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)
        self.group, self.config, self.pool, self.prefix = group, config, pool, prefix

    def _apply_quant_method(self, hidden_states, router_logits, input_ids):
        shared_output = self.shared_experts(hidden_states)
        topk_weights, topk_ids = self.router(hidden_states, router_logits, input_ids)
        fused_output = self.routed_experts(hidden_states, topk_weights, topk_ids)
        return shared_output, fused_output

    def _forward_impl(self, hidden_states, input_ids):
        router_logits, _ = self.gate(hidden_states)
        return self._apply_quant_method(hidden_states, router_logits, input_ids)

    def __call__(self, *, hidden_states, router_logits, input_ids):
        if router_logits is not hidden_states:
            raise ValueError("V4 Runner must own gate execution from its source hidden-state placeholder")
        shared_output, fused_output = self._forward_impl(hidden_states, input_ids)
        result = self.add(shared_output, fused_output)
        return self.group.call(self.prefix + ":all-reduce", result,
            pool_name=self.pool, config=self.config, inplace=False)


class DeepseekV4OutputProjection:
    """deep_gemm_fp8_o_proj source: inverse RoPE, grouped wo_a, row wo_b."""

    def __init__(self, trace, model, pool, prefix, group, config, rotary_cache, *, linear_source_profile):
        tp_size = len(group.traces)
        self.inverse_rope = SymbolicV4InverseRope(trace, prefix + ":inverse-rope", pool, model, tp_size=tp_size)
        self.wo_a = SymbolicV4GroupedOutput(trace, prefix + ":wo-a", pool, model, tp_size=tp_size)
        self.wo_b = SymbolicBlockFP8Linear(trace, prefix + ":wo-b", pool,
            model.o_groups // tp_size * model.o_lora_rank, model.hidden_size, source_profile=linear_source_profile)
        self.rotary_cache = rotary_cache
        self.group, self.config, self.pool, self.prefix = group, config, pool, prefix

    def __call__(self, o, positions):
        o_proj_input, o_scale = self.inverse_rope(o, positions, self.rotary_cache)
        z = self.wo_a(o_proj_input, o_scale)
        output, _ = self.wo_b(z.view_as_shape(z.shape[0], z.shape[1] * z.shape[2]))
        return self.group.call(self.prefix + ":all-reduce", output,
            pool_name=self.pool, config=self.config, inplace=False)


class DeepseekV4Attention:
    """Synchronous V4 FlashMLA forward with explicit scored-C4 applicability.

    C4's short-context path omits query/scoring, retaining both input
    projections and its compressor. C128 metadata tensors are explicit
    external forward inputs. Cache arenas are logical declarations, not a
    binding to the normal/native physical page allocator.
    """

    def __init__(self, trace, model, pool, prefix, group, config, rotary_cache, topk_buffer, *,
                 layer_index, q_lens, kv_lens, request_phases, slot_mapping,
                 cache_capacity_tokens, source_max_model_len, linear_source_profile, indexer_selection_source=None,
                 workspace=None, source_max_batched_tokens=None, cache_tensors=None, slot_tensor=None):
        from .runtime_inputs import dsv4_c128_metadata_width
        if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < 128
                or cache_capacity_tokens % 8 or cache_capacity_tokens < sum(kv_lens)):
            raise ValueError("V4 source logical arenas require explicit8-row capacity covering active contexts")
        self.trace, self.model, self.prefix = trace, model, prefix
        self.rotary_cache = rotary_cache
        self.local_heads = model.num_attention_heads // len(group.traces)
        self.padded_heads = 64 if self.local_heads <= 64 else 128
        ratio = max(1, model.compress_ratios[layer_index])
        roles = {"swa-cache"}
        if ratio > 1:
            roles.update(("partial-state", "compressed-cache"))
        if ratio == 4:
            roles.update(("indexer-partial-state", "indexer-cache"))
        if cache_tensors is not None and (not isinstance(cache_tensors, tuple)
                or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in cache_tensors)
                or len(cache_tensors) != len(roles) or {pair[0] for pair in cache_tensors} != roles):
            raise ValueError("V4 cache binding must provide exactly this layer's named persistent tensors")
        supplied = None if cache_tensors is None else dict(cache_tensors)
        caches = {}

        def cache(role, shape, dtype):
            tensor = (trace.tensor(prefix + ":" + role, shape, dtype, persistent=True)
                if supplied is None else supplied[role])
            trace._check_tensor(tensor)
            if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous:
                raise ValueError("V4 cache binding shape/dtype/layout differs from the declared arena")
            caches[role] = tensor
            return tensor

        long_c4 = ratio == 4 and max(kv_lens) // 4 > model.index_topk
        if long_c4 and indexer_selection_source != "declared_causal_topk":
            raise NotImplementedError("long C4 requires an explicit selected-cardinality declaration")
        self.inputs = DeepseekV4AttentionInput(trace, model, pool, prefix, layer_index=layer_index,
            linear_source_profile=linear_source_profile)
        self.wq_b = SymbolicBlockFP8Linear(trace, prefix + ":wq-b", pool,
            model.q_lora_rank, self.local_heads * model.head_dim, source_profile=linear_source_profile)
        self.swa_cache = cache("swa-cache", (cache_capacity_tokens, 584), "uint8")
        self.insert = SymbolicV4KVInsert(trace, prefix + ":swa-insert", pool, model,
            tp_size=len(group.traces), cache=self.swa_cache, slot_mapping=slot_mapping, slot_tensor=slot_tensor)
        self.compressor, self.indexer_compressor, self.short_topk = None, None, None
        self.indexer_wq_b, self.indexer_query, self.scored_topk = None, None, None
        compressed_cache = None
        if ratio > 1:
            coff, block = (2, 4) if ratio == 4 else (1, 8)
            partial = cache("partial-state", (cache_capacity_tokens // block, block,
                2 * coff * model.head_dim), "fp32")
            compressed_cache = cache("compressed-cache", (cache_capacity_tokens // ratio, 584), "uint8")
            self.compressor = SymbolicV4Compressor(trace, prefix + ":compressor", pool,
                head_dim=model.head_dim, compress_ratio=ratio, q_lens=q_lens, kv_lens=kv_lens,
                state_cache=partial, kv_cache=compressed_cache, graph_runtime_mode="NONE", slot_mapping=slot_mapping)
        if ratio == 4:
            index_partial = cache("indexer-partial-state",
                (cache_capacity_tokens // 4, 4, 4 * model.index_head_dim), "fp32")
            index_cache = cache("indexer-cache", (cache_capacity_tokens // 4, 132), "uint8")
            self.indexer_compressor = SymbolicV4Compressor(trace, prefix + ":indexer-compressor", pool,
                head_dim=model.index_head_dim, compress_ratio=4, q_lens=q_lens, kv_lens=kv_lens,
                state_cache=index_partial, kv_cache=index_cache, graph_runtime_mode="NONE", slot_mapping=slot_mapping)
            if long_c4:
                self.indexer_wq_b = SymbolicBlockFP8Linear(trace, prefix + ":indexer-wq-b", pool,
                    model.q_lora_rank, model.index_n_heads * model.index_head_dim, source_profile=linear_source_profile)
                self.indexer_query = SymbolicV4IndexerQuery(trace, prefix + ":indexer-query", pool, model)
                self.scored_topk = SymbolicV4ScoredTopK(trace, prefix + ":indexer-score-topk", pool, model,
                    topk_buffer, index_cache, q_lens=q_lens, kv_lens=kv_lens,
                    request_phases=request_phases, selection_source=indexer_selection_source,
                    workspace=workspace, source_max_model_len=source_max_model_len)
            else:
                self.short_topk = SymbolicV4ShortTopK(trace, prefix + ":short-topk", pool, topk_buffer)
        c128_indices = (None, None)
        if ratio == 128:
            capacity = dsv4_c128_metadata_width(max(kv_lens), source_max_model_len)
            # The metadata builder provides two distinct external input buffers.
            # Creating their descriptors does not price/build metadata on GPU.
            counts = tuple(sum(q for q, p in zip(q_lens, request_phases, strict=True) if p == phase)
                for phase in ("prefill", "decode"))
            c128_indices = tuple(None if count == 0 else trace.tensor(prefix + ":c128-" + phase + "-indices",
                (count, capacity), "int32", persistent=True)
                for phase, count in zip(("prefill", "decode"), counts, strict=True))
        self.attn = SymbolicV4SparseAttention(trace, prefix + ":sparse", pool, model,
            compress_ratio=ratio, q_lens=q_lens, kv_lens=kv_lens, request_phases=request_phases,
            swa_cache=self.swa_cache, compressed_cache=compressed_cache, topk_buffer=topk_buffer,
            c128_indices=c128_indices, source_max_model_len=source_max_model_len,
            indexer_selection_source=indexer_selection_source, workspace=workspace,
            source_max_batched_tokens=source_max_batched_tokens)
        self.o_proj = DeepseekV4OutputProjection(trace, model, pool, prefix + ":out", group, config,
            rotary_cache, linear_source_profile=linear_source_profile)
        self.cache_tensors = tuple(caches.items())

    def __call__(self, positions, hidden_states, llama_4_scaling=None):
        if llama_4_scaling is not None:
            raise NotImplementedError("V4 source profile does not consume llama4 scaling")
        rows = hidden_states.shape[0]
        o_padded = self.trace.tensor(self.trace.unique_name(self.prefix + ":o-padded"),
            (rows, self.padded_heads, self.model.head_dim), "bf16")
        qr_kv, kv_score, indexer_kv_score, indexer_weights = self.inputs._run_parallel_input_projections(hidden_states)
        qr, qr_scale, kv = self.inputs.norm(qr_kv)
        if qr_scale is not None:
            raise NotImplementedError("V4 SM90 source does not prequantize the q-lora norm result")
        self._prepare_and_attn(hidden_states, qr, kv, qr_scale, kv_score,
            indexer_kv_score, indexer_weights, positions, o_padded)
        o = o_padded.narrow(1, 0, self.local_heads)
        return self.o_proj(o, positions)

    def _prepare_and_attn(self, hidden_states, qr, kv, qr_scale, kv_score,
                          indexer_kv_score, indexer_weights, positions, o_padded):
        # The upstream helper owns Q/indexer outputs. Its return releases
        # those locals before forward enters the inverse-RoPE/output path.
        # The caller still owns qr/kv and the input projection results.
        rows = hidden_states.shape[0]
        q, _ = self.wq_b(qr)
        q = q.view_as_shape(rows, self.local_heads, self.model.head_dim)
        q = self.insert(q, kv, positions, self.rotary_cache)
        index_q, index_weights_out = None, None
        if self.indexer_wq_b is not None:
            index_q, _ = self.indexer_wq_b(qr)
            index_q = index_q.view_as_shape(rows, self.model.index_n_heads, self.model.index_head_dim)
            index_q, index_weights_out = self.indexer_query(index_q, indexer_weights, positions, self.rotary_cache)
        controls = []
        if self.indexer_compressor is not None:
            self.indexer_compressor(indexer_kv_score, positions, self.rotary_cache)
            if self.short_topk is not None:
                self.short_topk(positions, controls=(self.indexer_compressor.kv_cache,))
            controls.append(self.indexer_compressor.state_cache)
        if self.compressor is not None:
            self.compressor(kv_score, positions, self.rotary_cache)
            controls.append(self.compressor.state_cache)
        if self.scored_topk is not None:
            self.scored_topk(index_q, index_weights_out,
                controls=(self.swa_cache, self.compressor.kv_cache, *controls))
        self.attn(q, positions, o_padded, controls=tuple(controls))


class DeepseekV4ProfileAttention(DeepseekV4Attention):
    """The same forward with vLLM's explicit no-metadata profile helpers.

    This is a storage program: no real cache/selector/attention execution,
    no timing claim. Query/output projections still execute symbolically.
    Workspace and dummy logits are distinct source allocations, never an
    inferred correction to an observed activation peak.
    """

    def __init__(self, trace, model, pool, prefix, group, config, rotary_cache, topk_buffer, *,
                 layer_index, workspace, source_max_model_len, source_max_batched_tokens,
                 linear_source_profile, indexer_max_logits_mb):
        from .traced_sampling import StorageOnlyOps
        if (not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace
                or any(type(value) is not int or value <= 0 for value in (
                    source_max_model_len, source_max_batched_tokens, indexer_max_logits_mb))):
            raise ValueError("V4 profile requires explicit rank workspace, context/batch limits and indexer logits MiB")
        self.trace, self.model, self.prefix = trace, model, prefix
        self.local_heads = model.num_attention_heads // len(group.traces)
        self.padded_heads = 64 if self.local_heads <= 64 else 128
        self.ratio = max(1, model.compress_ratios[layer_index])
        self.workspace, self.max_model_len = workspace, source_max_model_len
        self.max_batched_tokens, self.indexer_max_logits_mb = source_max_batched_tokens, indexer_max_logits_mb
        self.rotary_cache, self.topk_buffer = rotary_cache, topk_buffer
        self.ops = StorageOnlyOps(trace, prefix + ":profile", pool)
        self.inputs = DeepseekV4AttentionInput(trace, model, pool, prefix, layer_index=layer_index,
            linear_source_profile=linear_source_profile)
        self.wq_b = SymbolicBlockFP8Linear(trace, prefix + ":wq-b", pool,
            model.q_lora_rank, self.local_heads * model.head_dim, source_profile=linear_source_profile)
        self.indexer_wq_b, self.indexer_query = None, None
        if self.ratio == 4:
            self.indexer_wq_b = SymbolicBlockFP8Linear(trace, prefix + ":indexer-wq-b", pool,
                model.q_lora_rank, model.index_n_heads * model.index_head_dim, source_profile=linear_source_profile)
            self.indexer_query = SymbolicV4IndexerQuery(trace, prefix + ":indexer-query", pool, model)
        self.o_proj = DeepseekV4OutputProjection(trace, model, pool, prefix + ":out", group, config,
            rotary_cache, linear_source_profile=linear_source_profile)

    def _profile_compressor(self, kv_score, head_dim):
        # compressor.py splits views, then returns before APE/save/finish.
        coff = 2 if self.ratio == 4 else 1
        kv, score = kv_score.split((coff * head_dim, coff * head_dim), dim=-1)

    def _profile_indexer_inputs(self, qr, indexer_weights, indexer_kv_score, positions):
        q, _ = self.indexer_wq_b(qr)
        q = q.view_as_shape(q.shape[0], self.model.index_n_heads, self.model.index_head_dim)
        result = self.indexer_query(q, indexer_weights, positions, self.rotary_cache)
        self._profile_compressor(indexer_kv_score, self.model.index_head_dim)
        return result

    def _profile_indexer(self, hidden_states, q_quant, weights):
        # sparse_attn_indexer.py: no-metadata get_simultaneous result is
        # discarded; its arena persists via the manager. The UINT8 dummy
        # allocation is local to this call, not the whole model forward.
        capacity = self.max_model_len * 40 // 4
        self.workspace.get_simultaneous(((capacity, self.model.index_head_dim), "fp8"),
            ((capacity, 4), "uint8"), ((1024 * 1024,), "uint8"))
        dummy_logits = self.trace.tensor(self.trace.unique_name(self.prefix + ":profile-logits"),
            (self.indexer_max_logits_mb * 1024 * 1024,), "uint8")
        return self.topk_buffer

    def _profile_sparse_attention(self, q, o_padded):
        # nvidia/flashmla.py reserves the maximum dummy-prefill ports,
        # independently of the active request lengths, then zeroes output.
        if q.shape != o_padded.shape or q.dtype != o_padded.dtype:
            raise ValueError("V4 profile Q and zeroed output require matching padded shape/dtype")
        n = 0 if self.ratio == 1 else -(-self.max_model_len // self.ratio)
        width = n + self.model.window_size + self.max_batched_tokens
        topk = 0 if self.ratio == 1 else self.topk_buffer.shape[-1]
        combined = -(-(topk + self.model.window_size) // 128) * 128
        self.workspace.get_simultaneous(((4, width, self.model.head_dim), "bf16"),
            ((self.max_batched_tokens, combined), "int32"), ((self.max_batched_tokens,), "int32"))
        self.ops.operation("no_metadata_flashmla_zero_output", (),
            shape=o_padded.shape, dtype=o_padded.dtype, out=o_padded)

    def _prepare_and_attn(self, hidden_states, qr, kv, qr_scale, kv_score,
                          indexer_kv_score, indexer_weights, positions, o_padded):
        q, _ = self.wq_b(qr)
        q = q.view_as_shape(q.shape[0], self.local_heads, self.model.head_dim)
        if self.local_heads < self.padded_heads:
            q = self.ops.unary("no_metadata_pad_query_heads", q,
                shape=(q.shape[0], self.padded_heads, self.model.head_dim))
        index_q, index_weights_out = None, None
        if self.indexer_wq_b is not None:
            index_q, index_weights_out = self._profile_indexer_inputs(qr, indexer_weights,
                indexer_kv_score, positions)
        if self.ratio > 1:
            self._profile_compressor(kv_score, self.model.head_dim)
        if index_q is not None:
            self._profile_indexer(hidden_states, index_q, index_weights_out)
        self._profile_sparse_attention(q, o_padded)


class DeepseekV4DecoderLayer:
    """PP1/non-MegaMoE DeepseekV4DecoderLayer.forward on virtual modules."""

    def __init__(self, trace, model, pool, prefix, group, config, rotary_cache, topk_buffer, *,
                 layer_index, q_lens, kv_lens, request_phases, slot_mapping, cache_capacity_tokens,
                 source_max_model_len, linear_source_profile, source_use_deep_gemm, hardware,
                 resolution, invocation_id, indexer_selection_source=None, workspace=None, source_max_batched_tokens=None,
                 profile_indexer_max_logits_mb=None, ffn=None, mhc_workspace_scope="source_launch", cache_tensors=None, slot_tensor=None):
        self.pre = SymbolicMHC(trace, prefix + ":pre", pool, model, hardware,
            variant="mhc_pre", source_use_deep_gemm=source_use_deep_gemm, workspace_scope=mhc_workspace_scope)
        self.attn_pre = SymbolicMHC(trace, prefix + ":attn-post-pre", pool, model, hardware,
            variant="mhc_post_pre", source_use_deep_gemm=source_use_deep_gemm, workspace_scope=mhc_workspace_scope)
        self.ffn_pre = SymbolicMHC(trace, prefix + ":ffn-post-pre", pool, model, hardware,
            variant="mhc_post_pre", source_use_deep_gemm=source_use_deep_gemm, workspace_scope=mhc_workspace_scope)
        if profile_indexer_max_logits_mb is None:
            self.attn = DeepseekV4Attention(trace, model, pool, prefix + ":attn", group, config, rotary_cache, topk_buffer,
                layer_index=layer_index, q_lens=q_lens, kv_lens=kv_lens, request_phases=request_phases,
                slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
                source_max_model_len=source_max_model_len, linear_source_profile=linear_source_profile,
                indexer_selection_source=indexer_selection_source, workspace=workspace,
                source_max_batched_tokens=source_max_batched_tokens, cache_tensors=cache_tensors, slot_tensor=slot_tensor)
        else:
            if cache_tensors is not None:
                raise ValueError("no-metadata profile does not consume an existing request cache")
            self.attn = DeepseekV4ProfileAttention(trace, model, pool, prefix + ":attn", group, config,
                rotary_cache, topk_buffer, layer_index=layer_index, workspace=workspace,
                source_max_model_len=source_max_model_len, source_max_batched_tokens=source_max_batched_tokens,
                linear_source_profile=linear_source_profile, indexer_max_logits_mb=profile_indexer_max_logits_mb)
        self.ffn = ffn if ffn is not None else DeepseekV4MoE(trace, model, pool, prefix + ":ffn", group, config,
            layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
            linear_source_profile=linear_source_profile, workspace=workspace,
            source_multiprocessors=int(hardware.num_compute_units))

    def prepare_ffn(self, x, positions, post_mix=None, res_mix=None, residual=None):
        """Trace attention and return the exact FFN inputs for staged execution."""
        if residual is None:
            residual, post_mix, res_mix, x = self.pre(x)
        else:
            residual, post_mix, res_mix, x = self.attn_pre(x, residual, post_mix, res_mix)
        x = self.attn(positions, x, None)
        residual, post_mix, res_mix, x = self.ffn_pre(x, residual, post_mix, res_mix)
        return x, residual, post_mix, res_mix

    def __call__(self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None):
        x, residual, post_mix, res_mix = self.prepare_ffn(
            x, positions, post_mix, res_mix, residual)
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix


class DeepseekV4Model:
    """Complete PP1/MTP-off V4 layer loop; source metadata/provider are explicit."""

    def __init__(self, trace, model, pool, group, config, *, token_ids, q_lens, kv_lens, request_phases,
                 slot_mapping, cache_capacity_tokens, source_max_model_len, max_batched_tokens,
                 linear_source_profile, source_use_deep_gemm, hardware, resolutions, invocation_ids, model_input_order,
                 indexer_selection_source=None, workspace_initial_capacity_bytes=None, profile_indexer_max_logits_mb=None, slot_tensor=None):
        self.embed_tokens = SymbolicVocabParallelEmbedding(trace, "embedding", pool, model.hidden_size,
            model.vocab_size, group, config, token_ids=token_ids,
            model_input_order=model_input_order if trace.rank == group.participants[0] else None)
        topk_buffer = trace.tensor("model.topk-buffer", (max_batched_tokens, model.index_topk), "int32", persistent=True)
        # finalize_mhc_broadcast_weights retains the first layer's reduced
        # FP32 weight buffer. Ordinary parameters are accounted separately;
        # this derived model-owned buffer belongs to the source residency.
        self.hc_attn_fn_broadcast = trace.tensor("model.first-layer-mhc-broadcast",
            (model.hc_mult * (model.hc_mult + 2), model.hidden_size), "fp32", persistent=True)
        # build_deepseek_v4_rope/get_rope shares equal constructor keys. C4
        # and C128 use the same compressed-table key; the indexer borrows it.
        rope_kinds = tuple(dict.fromkeys("swa" if ratio <= 1 else "compressed"
            for ratio in model.compress_ratios[:model.num_hidden_layers]))
        ropes = {kind: trace.tensor("model.rope-" + kind,
            (model.max_position_embeddings, model.qk_rope_head_dim), "fp32", persistent=True) for kind in rope_kinds}
        self.layers = []
        workspace = (None if workspace_initial_capacity_bytes is None else
            ForwardWorkspace(trace, "worker.workspace", alignment_bytes=256,
                initial_capacity_bytes=workspace_initial_capacity_bytes, locked=False))
        for layer in range(model.num_hidden_layers):
            rope = ropes["swa" if model.compress_ratios[layer] <= 1 else "compressed"]
            self.layers.append(DeepseekV4DecoderLayer(trace, model, pool, f"layer-{layer}", group, config,
                rope, topk_buffer, layer_index=layer, q_lens=q_lens, kv_lens=kv_lens,
                request_phases=request_phases, slot_mapping=slot_mapping, slot_tensor=slot_tensor, cache_capacity_tokens=cache_capacity_tokens,
                source_max_model_len=source_max_model_len, linear_source_profile=linear_source_profile,
                source_use_deep_gemm=source_use_deep_gemm, hardware=hardware,
                resolution=resolutions[layer], invocation_id=invocation_ids[layer],
                indexer_selection_source=indexer_selection_source, workspace=workspace,
                source_max_batched_tokens=max_batched_tokens, profile_indexer_max_logits_mb=profile_indexer_max_logits_mb))
        self.post = SymbolicMHC(trace, "final-mhc-post", pool, model, hardware,
            variant="mhc_post", source_use_deep_gemm=source_use_deep_gemm)
        self.head = SymbolicMHC(trace, "hc-head", pool, model, hardware,
            variant="hc_head", source_use_deep_gemm=source_use_deep_gemm)
        self.norm = SymbolicRMSNorm(trace, "final-norm", pool, model.hidden_size, model.target_semantics.rms_norm_eps)

    def __call__(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual, post_mix, res_mix = None, None, None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(hidden_states, positions, input_ids,
                post_mix, res_mix, residual)
        hidden_states = self.post(hidden_states, residual, post_mix, res_mix)
        hidden_states = self.head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


def trace_deepseek_v4_forward(model, step, pool, hardware, expert_routing, *, input_ids, positions,
                             slot_mapping, cache_capacity_tokens, source_max_model_len, max_batched_tokens,
                             source_use_deep_gemm, linear_source_profile, source_prompt_lengths=None,
                             indexer_selection_source=None, workspace_initial_capacity_bytes=None, sample_logits=True,
                             cast_logits=False):
    """Complete metadata-only V4 target forward and greedy chain, TP4/8 PP1.

    Requests and token vectors are packed by the shared source metadata program.
    Routing is an explicit normalized dynamic input; no checkpoint assignment
    or token values are computed. Native allocator/metadata services and
    workspace gaps remain separate from this declared model-forward scope.
    """
    from src.serving.schedule.schedule_step import schedule_step_sha256
    from .runtime_inputs import vllm_0_30_source_model_inputs
    from .runtime_inputs import ExpertRoutingInput
    if type(sample_logits) is not bool or type(cast_logits) is not bool or sample_logits and cast_logits:
        raise TypeError("V4 output flags must be exclusive booleans")
    if indexer_selection_source not in (None, "declared_causal_topk"):
        raise ValueError("unsupported symbolic V4 indexer selection source")
    config = pool.parallel_plan.config
    if workspace_initial_capacity_bytes is not None and (
            not isinstance(workspace_initial_capacity_bytes, tuple)
            or len(workspace_initial_capacity_bytes) != config.tensor_parallel_size
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("V4 source workspace history needs one explicit nonnegative entry capacity per physical rank")
    if (model.model_type != "deepseek_v4" or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
            or pool.hardware_id != "nvidia_h200" or config.tensor_parallel_size not in (4, 8)
            or config.enable_expert_parallel or any(value != 1 for value in (
                config.pipeline_parallel_size, config.sequence_parallel_size, config.context_parallel_size,
                config.data_parallel_size))):
        raise NotImplementedError("symbolic V4 target source requires H200 block-FP8/BF16 TP4/8 PP/SP/CP/DP1 EP-off")
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("symbolic V4 forward requires explicit canonical routing input")
    if any(r.mtp_proposed_tokens or r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests):
        raise NotImplementedError("symbolic V4 source currently requires MTP-off")
    source_step = step
    rows = sum(r.scheduled_tokens for r in step.requests)
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(v) is not int for v in values):
            raise ValueError(f"symbolic V4 {name} requires one explicit integer per model row")
    if any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("symbolic V4 input IDs must belong to the accurate checkpoint vocabulary")
    if (type(source_max_model_len) is not int
            or not max(r.computed_tokens + r.scheduled_tokens for r in step.requests)
                <= source_max_model_len <= model.max_position_embeddings):
        raise ValueError("symbolic V4 source maximum length must contain every declared context")
    expected_positions = tuple(p for r in step.requests for p in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens))
    if positions != expected_positions or any(value >= source_max_model_len for value in positions):
        raise ValueError("symbolic V4 positions must agree with explicit request contexts and maximum length")
    if type(max_batched_tokens) is not int or max_batched_tokens < rows:
        raise ValueError("symbolic V4 batch capacity must cover all scheduled rows")
    if type(source_use_deep_gemm) is not bool:
        raise TypeError("symbolic V4 mHC needs an explicit source DeepGEMM branch")
    if len(set(slot_mapping)) != rows:
        raise ValueError("symbolic V4 source requires distinct writable logical slots")
    step, input_ids, model_input_order = vllm_0_30_source_model_inputs(step,
        embedding_token_ids=input_ids, prompt_lengths=source_prompt_lengths)
    positions = tuple(positions[row] for row in model_input_order["token_row_permutation"])
    slot_mapping = tuple(slot_mapping[row] for row in model_input_order["token_row_permutation"])
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    phases = tuple(model_input_order["model_attention_phases"])
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    collective = pool.collective_for("all_reduce")
    gather = pool.collective_for("all_gather")
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    step_digest = schedule_step_sha256(source_step)
    resolutions = tuple(expert_routing.resolve(step_id=step.step_id, layer_index=layer,
        moe_invocation_id=invocations[layer], schedule_step_sha256=step_digest,
        tokens=rows, num_experts=model.num_experts, topk=model.num_experts_per_tok,
        model_config_sha256=model.config_sha256, pool_name=pool.name,
        tensor_parallel_size=config.tensor_parallel_size, expert_parallel_size=1,
        expert_rank_group=(pool.ranks[0],), expert_placement_strategy="linear",
        moe_layer_ordinal=model.moe_layer_ordinal(layer)) for layer in range(model.num_hidden_layers))
    selected_rows = tuple(sum(q_lens[:i + 1]) - 1 for i in range(len(q_lens)))
    def rank_forward(trace, workspace_capacity):
        slots = trace.tensor("slot_mapping", (rows,), "int64")
        network = DeepseekV4Model(trace, model, pool.name, group, collective,
            token_ids=input_ids, q_lens=q_lens, kv_lens=kv_lens, request_phases=phases,
            slot_mapping=slot_mapping, slot_tensor=slots, cache_capacity_tokens=cache_capacity_tokens,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            linear_source_profile=linear_source_profile, source_use_deep_gemm=source_use_deep_gemm,
            hardware=hardware, resolutions=resolutions, invocation_ids=invocations, model_input_order=model_input_order,
            indexer_selection_source=indexer_selection_source, workspace_initial_capacity_bytes=workspace_capacity)
        ids = trace.tensor("input_ids", (rows,), "int32")
        pos = trace.tensor("positions", (rows,), "int64")
        indices = trace.tensor("logits_indices", (len(selected_rows),), "int32")
        hidden_states = network(ids, pos)
        selected = SymbolicLogitsRowSelect(trace, "logits-row-select", pool.name)(hidden_states,
            indices, selected_indices=selected_rows)
        processor = SymbolicTPLogitsProcessor(trace, "lm-head", pool.name, hidden_size=model.hidden_size,
            vocab_size=model.vocab_size, group=group, config=gather)
        logits = processor(selected)
        if sample_logits or cast_logits:
            sampled = SymbolicGreedy(trace, "target-sampler", pool.name)(logits, sample_logits=sample_logits)
        return None

    for ordinal, trace in enumerate(traces):
        rank_forward(trace, None if workspace_initial_capacity_bytes is None else workspace_initial_capacity_bytes[ordinal])
    return group


def trace_deepseek_v4_startup_profile(model, pool, hardware, expert_routing, *, tokens, max_requests,
        source_max_model_len, source_use_deep_gemm, workspace_initial_capacity_bytes,
        indexer_max_logits_mb, source_profile):
    """Complete V4 layer loop plus shared worker/sampler storage scopes.

    Explicit eager/no-attention-metadata, synchronous modular TP, pure text,
    EP/SP/CP/MTP-off and native PyTorch-sort/FP32-noise sampler. This is not
    a normal request graph or a timed program. No device tensor is created.
    """
    from .runtime_inputs import ExpertRoutingInput
    from .traced_sampling import VLLM030WorkerStorage, worker_profile_storage_forward
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
    config = pool.parallel_plan.config
    if (source_profile != "vllm_0_30_v4_sync_no_metadata_pytorch_sort_fp32_noise"
            or model.model_type != "deepseek_v4" or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
            or pool.hardware_id != "nvidia_h200" or hardware.hardware_id != pool.hardware_id
            or (model.model_id, config.tensor_parallel_size) not in (("deepseek-ai/DeepSeek-V4-Flash", 4),
                ("deepseek-ai/DeepSeek-V4-Pro", 8)) or config.enable_expert_parallel
            or any(value != 1 for value in (config.pipeline_parallel_size, config.sequence_parallel_size,
                config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("V4 startup storage requires explicit H200 TP4/8 synchronous no-metadata source")
    if (any(type(value) is not int or value <= 0 for value in (tokens, max_requests, source_max_model_len,
                indexer_max_logits_mb)) or source_max_model_len > model.max_position_embeddings
            or type(source_use_deep_gemm) is not bool):
        raise ValueError("V4 startup requires positive explicit limits/logits MiB and a declared mHC source branch")
    if (not isinstance(workspace_initial_capacity_bytes, tuple)
            or len(workspace_initial_capacity_bytes) != config.tensor_parallel_size
            or any(type(value) is not int or value < 0 for value in workspace_initial_capacity_bytes)):
        raise ValueError("V4 startup requires explicit workspace entry history per physical rank")
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("V4 startup routing must be a canonical explicit dynamic input")
    rows = min(tokens, max_requests)
    lengths = (tokens // rows,) * (rows - 1) + (tokens // rows + tokens % rows,)
    if max(lengths) > source_max_model_len:
        raise ValueError("V4 startup maximum length must cover each dummy request, not the sum of batch tokens")
    step = ScheduleStep("startup-profile", pool.name, tuple(
        ScheduledRequest(str(index), "prefill", length, 0) for index, length in enumerate(lengths)))
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, tokens)
    traces = tuple(ForwardTrace(rank=rank, namespace=f"profile:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    collective, gather = pool.collective_for("all_reduce"), pool.collective_for("all_gather")

    def rank_profile(trace, capacity):
        network = DeepseekV4Model(trace, model, pool.name, group, collective, token_ids=(0,) * tokens,
            q_lens=lengths, kv_lens=lengths, request_phases=("prefill",) * rows,
            slot_mapping=None, cache_capacity_tokens=None, source_max_model_len=source_max_model_len,
            max_batched_tokens=tokens, linear_source_profile="vllm_0_30_row_major_group128_fp32",
            source_use_deep_gemm=source_use_deep_gemm, hardware=hardware,
            resolutions=resolutions, invocation_ids=invocations, model_input_order=None,
            workspace_initial_capacity_bytes=capacity, profile_indexer_max_logits_mb=indexer_max_logits_mb)
        worker_storage = VLLM030WorkerStorage(trace, tokens=tokens, max_requests=max_requests,
            inputs_embeds_size=model.hidden_size, dtype=model.compute_dtype)
        head = SymbolicTPLogitsProcessor(trace, "profile.lm-head", pool.name, hidden_size=model.hidden_size,
            vocab_size=model.vocab_size, group=group, config=gather)
        trace.checkpoint("worker.profile.begin")
        worker_profile_storage_forward(trace, network, head,
            input_ids=worker_storage.input_ids, positions=worker_storage.positions,
            query_lengths=lengths, sampler_profile="vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise",
            pool_name=pool.name)
        trace.checkpoint("worker.profile.end")

    for trace, capacity in zip(traces, workspace_initial_capacity_bytes, strict=True):
        rank_profile(trace, capacity)
    return group


class DeepseekV4MoE:
    """V4's non-MegaMoE forward specialization; its Runner owns GateLinear."""

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index,
                 resolution, invocation_id, linear_source_profile, workspace=None, source_multiprocessors=None):
        if model.model_type != "deepseek_v4" or model.num_shared_experts != 1:
            raise NotImplementedError("symbolic V4 modular source currently requires its one shared expert")
        self.hash_routing = layer_index < model.num_hash_layers
        self.experts = DeepseekV4MoERunner(trace, model, pool, prefix, group, config,
            layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
            linear_source_profile=linear_source_profile, workspace=workspace,
            source_multiprocessors=source_multiprocessors)

    def __call__(self, hidden_states, input_ids=None):
        if self.hash_routing and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids")
        return self._forward_fused_moe(hidden_states, input_ids)

    def _forward_fused_moe(self, hidden_states, input_ids=None):
        org_shape = hidden_states.shape
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            input_ids=input_ids,
        )
        return final_hidden_states.view_as_shape(org_shape)


class Qwen3Attention:
    qk_norm = True

    def __init__(self, trace, model, pool, prefix, rope_cache, kv_cache, q_lens, kv_lens, slot_mapping,
                 *, profile_no_metadata=False, group=None, collective=None, rope_program=ROPE_PROGRAM_FUSED,
                 hardware_id="nvidia_h200", collective_inplace=False,
                 model_execution_version="vllm_0_30_0"):
        tp_size = 1 if group is None else len(group.traces)
        if model.num_attention_heads % tp_size:
            raise ValueError("query head count must divide the TP group")
        heads = model.num_attention_heads // tp_size
        kv_heads = model.kv_heads_per_tensor_parallel_rank(tp_size)
        self.head_dim = model.head_dim
        self.q_size = heads * model.head_dim
        self.kv_size = kv_heads * model.head_dim
        self.qkv_proj = SymbolicLinear(trace, prefix + ":qkv", pool,
            model.hidden_size, self.q_size + 2 * self.kv_size, bias=model.attention_bias)
        if self.qk_norm:
            norm_program = ("vllm_ascend_cann" if hardware_id == "ascend_910b" else
                "vllm_c_cuda_0_11" if model_execution_version == "vllm_0_11_0" else "vllm_c_cuda")
            self.q_norm = SymbolicRMSNorm(trace, prefix + ":q-norm", pool, model.head_dim, model.rms_norm_eps,
                storage_program=norm_program)
            self.k_norm = SymbolicRMSNorm(trace, prefix + ":k-norm", pool, model.head_dim, model.rms_norm_eps,
                storage_program=norm_program)
        self.rotary_emb = SymbolicRotary(trace, prefix + ":rotary", pool,
            heads, kv_heads, model.head_dim, rope_cache, program=rope_program,
            rotary_dim=model.rotary_dim,
            operand_semantics="aclnn_materialized_operands" if hardware_id == "ascend_910b" else "aten_tensor_iterator")
        self.attn = SymbolicAttention(trace, prefix + ":attention", pool,
            heads, kv_heads, model.head_dim, kv_cache, q_lens, kv_lens, slot_mapping,
            profile_no_metadata=profile_no_metadata, model_execution_version=model_execution_version)
        self.o_proj = SymbolicLinear(trace, prefix + ":o-proj", pool, self.q_size, model.hidden_size)
        if group is not None:
            self.o_proj = SymbolicRowParallelLinear(self.o_proj, group, collective, inplace=collective_inplace)

    def __call__(self, positions, hidden_states):
        # Qwen3Attention.forward; only Tensor.view's spelling is adapted.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view_as_shape(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view_as_shape(q.shape)
        k_by_head = k.view_as_shape(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view_as_shape(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaAttention(Qwen3Attention):
    """vLLM LlamaAttention.forward, sharing QKV/RoPE/cache primitives."""

    qk_norm = False

    def __call__(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer:
    attention_type = Qwen3Attention
    mlp_type = QwenMLP

    def __init__(self, trace, model, pool, prefix, rope_cache, kv_cache, q_lens, kv_lens, slot_mapping,
                 *, profile_no_metadata=False, group=None, collective=None, mlp_factory=None,
                 rope_program=ROPE_PROGRAM_FUSED, hardware_id="nvidia_h200", ffn_factory=None, collective_inplace=False,
                 model_execution_version="vllm_0_30_0"):
        self._source_prefix = prefix
        norm_program = ("vllm_ascend_cann" if hardware_id == "ascend_910b" else
            "vllm_c_cuda_0_11" if model_execution_version == "vllm_0_11_0" else "vllm_c_cuda")
        self.input_layernorm = SymbolicRMSNorm(trace, prefix + ":input-norm", pool, model.hidden_size, model.rms_norm_eps,
            storage_program=norm_program)
        self.self_attn = self.attention_type(trace, model, pool, prefix, rope_cache, kv_cache, q_lens, kv_lens, slot_mapping,
            profile_no_metadata=profile_no_metadata, group=group, collective=collective, rope_program=rope_program,
            hardware_id=hardware_id, collective_inplace=collective_inplace,
            model_execution_version=model_execution_version)
        self.remote_ffn = None
        if ffn_factory is None:
            self.post_attention_layernorm = SymbolicRMSNorm(trace, prefix + ":post-attn-norm", pool, model.hidden_size, model.rms_norm_eps,
                storage_program=norm_program)
            factory = self.mlp_type if mlp_factory is None else mlp_factory
            self.mlp = factory(trace, model, pool, prefix, group=group, collective=collective,
                **({"collective_inplace": True} if collective_inplace else {}))
        else:
            self.remote_ffn = ffn_factory(trace, model, pool, prefix,
                mlp_type=self.mlp_type if mlp_factory is None else mlp_factory)

    def __call__(self, positions, hidden_states, residual):
        # Qwen3DecoderLayer.forward, retaining source alias/caller lifetimes.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        if self.remote_ffn is not None:
            return self.remote_ffn(hidden_states, residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaDecoderLayer(Qwen3DecoderLayer):
    attention_type = LlamaAttention
    mlp_type = LlamaMLP


class RemoteFFN:
    """Existing AF placement around the shared source norm/MLP modules."""

    def __init__(self, trace, model, pool, prefix, *, mlp_type, attention_pool, ffn_pool, transport, group):
        lane = attention_pool.ranks.index(trace.rank)
        destination = transport.pool_traces[ffn_pool.name][lane]
        self.transport, self.attention_pool, self.ffn_pool, self.prefix = transport, pool, ffn_pool.name, prefix
        self.invocations = 0
        self.norm = SymbolicRMSNorm(destination, prefix + ":post-attn-norm", ffn_pool.name,
            model.hidden_size, model.rms_norm_eps, storage_program="vllm_c_cuda")
        self.mlp = mlp_type(destination, model, ffn_pool.name, prefix, group=group,
            collective=None if group is None else ffn_pool.collective_for("all_reduce"))

    def __call__(self, hidden_states, residual):
        boundary = self.prefix if self.invocations == 0 else f"{self.prefix}:repeat-{self.invocations}"
        self.invocations += 1
        self.transport.end_local_capture(hidden_states.trace,
            {"hidden_states": hidden_states, "residual": residual})
        hidden_states, residual = self.transport.exchange(hidden_states, residual,
            source_pool=self.attention_pool, destination_pool=self.ffn_pool, boundary=boundary + ":to-ffn")
        self.transport.begin_local_capture(hidden_states.trace,
            {"hidden_states": hidden_states, "residual": residual}, boundary=boundary + ":ffn")
        hidden_states, residual = self.norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        self.transport.end_local_capture(hidden_states.trace,
            {"hidden_states": hidden_states, "residual": residual})
        hidden_states, residual = self.transport.exchange(hidden_states, residual, source_pool=self.ffn_pool,
            destination_pool=self.attention_pool, boundary=boundary + ":to-attention")
        self.transport.begin_local_capture(hidden_states.trace,
            {"hidden_states": hidden_states, "residual": residual}, boundary=boundary + ":attention")
        return hidden_states, residual


def _dense_final_norm(trace, model, pool, hardware_id):
    return SymbolicRMSNorm(trace, "final-norm", pool, model.hidden_size, model.rms_norm_eps,
        storage_program="vllm_ascend_cann" if hardware_id == "ascend_910b" else "vllm_c_cuda")


def _ascend018_interleaved_rope_constructor_views(trace, cache):
    """Retain the plugin's one expanded backing, not two extra RoPE inputs.

    AscendRotaryEmbedding.__init__ records module-global cos/sin views via
    _record_cos_and_sin_cache_interleaved. Its repeat creates one allocation;
    chunk/squeeze retain aliases. Ordinary GQA rope_forward_oot still consumes
    the original cos_sin_cache, so these constructor buffers add no read ports.
    They are persistent model buffers, not parameter or activation payloads.
    """
    trace._check_tensor(cache)
    if (cache.dtype != "bf16" or len(cache.shape) != 2 or not cache.is_contiguous
            or cache.shape[1] % 2):
        raise ValueError("Ascend0.18 interleaved constructor requires dense BF16 even-width RoPE cache")
    positions, width = cache.shape
    backing = trace.tensor("model.rope-interleaved-cache", (positions, 2, width), cache.dtype, persistent=True)
    strides = (backing.view.strides_bytes[0], backing.view.strides_bytes[2])
    cos = trace.alias(backing, "model.rope-cos-cache", shape=cache.shape, strides_bytes=strides)
    sin = trace.alias(backing, "model.rope-sin-cache", shape=cache.shape, strides_bytes=strides,
        offset_bytes=width * backing.view.element_bytes)
    return cos, sin


class Qwen3Model:
    decoder_type = Qwen3DecoderLayer
    model_execution_version = "vllm_0_30_0"
    # Derived models may build their own source layers while inheriting forward.
    # Only the homogeneous constructor below registers a replay program.
    _source_module_program = None

    def __init__(self, trace, model, pool, q_lens, kv_lens, cache_capacity_tokens, slot_mapping,
                 *, profile_no_metadata=False, group=None, collective=None, token_ids=None, mlp_factories=None,
                 fused_embedding=True, rope_program=ROPE_PROGRAM_FUSED, hardware_id="nvidia_h200", layer_range=None,
                 ffn_factory=None, input_mode="token_ids", model_execution_version=None,
                 maca_copy_runtime_source=None):
        self.trace = trace
        self.model_execution_version = (type(self).model_execution_version if model_execution_version is None else model_execution_version)
        if self.model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
            raise ValueError("dense forward requires its explicit common model source version")
        if maca_copy_runtime_source is not None and (
                hardware_id != "metax_c500" or self.model_execution_version != "vllm_0_24_0"
                or group is not None or model.num_experts):
            raise NotImplementedError("MACA Copy runtime source requires ordinary C5000.24 dense TP1")
        if input_mode not in ("token_ids", "inputs_embeds"):
            raise ValueError("model input_mode requires token_ids or inputs_embeds")
        if type(profile_no_metadata) is not bool:
            raise TypeError("model attention profile must be explicit boolean metadata")
        tp_size = 1 if group is None else len(group.traces)
        if layer_range is None:
            layer_range = (0, model.num_hidden_layers)
        if (not isinstance(layer_range, tuple) or len(layer_range) != 2
                or any(type(index) is not int for index in layer_range)
                or not 0 <= layer_range[0] < layer_range[1] <= model.num_hidden_layers):
            raise ValueError("model stage requires an explicit nonempty decoder layer range")
        self.first_stage, self.last_stage = layer_range[0] == 0, layer_range[1] == model.num_hidden_layers
        self._source_module_program = (trace._module_program(self.decoder_type)
            if group is None and mlp_factories is None and ffn_factory is None and not model.num_experts
                and model.compute_dtype == model.weight_dtype and model.compute_dtype in ("bf16", "fp16")
                and not profile_no_metadata and self.decoder_type in (Qwen3DecoderLayer, LlamaDecoderLayer)
            else None)
        self.inputs_embeds_shape = (sum(q_lens), model.hidden_size)
        self.inputs_embeds_dtype = model.compute_dtype
        ascend018 = self.model_execution_version == "vllm_0_18_0"
        if ascend018 and (hardware_id != "ascend_910b" or model.compute_dtype != "bf16"
                or model.weight_dtype != "bf16" or ffn_factory is not None
                or (model.num_experts and (model.model_type != "qwen3_moe" or mlp_factories is None or tp_size != 2))):
            raise NotImplementedError("Ascend0.18 ordinary model requires unquantized BF16 dense source")
        self.embed_tokens = None
        if self.first_stage and input_mode == "token_ids":
            self.embed_tokens = (SymbolicEmbedding(trace, "embedding", pool, model.hidden_size, model.compute_dtype,
                model_execution_version=self.model_execution_version, vocab_size=model.vocab_size,
                maca_copy_runtime_source=maca_copy_runtime_source)
            if group is None else SymbolicVocabParallelEmbedding(trace, "embedding", pool,
                model.hidden_size, model.vocab_size, group, collective, token_ids=token_ids, fused=fused_embedding,
                dtype=model.compute_dtype, model_execution_version=self.model_execution_version,
                reduction_inplace=ascend018,
                mask_program=("vllm_0_11" if self.model_execution_version == "vllm_0_11_0" else
                    "vllm_0_24" if self.model_execution_version == "vllm_0_24_0" else
                    "vllm_ascend_no_added_vocab" if hardware_id == "ascend_910b" else "vllm_0_30")))
        # get_rope caches one module shared by identical layer constructors.
        self.rope_cache = trace.tensor("model.rope-cache",
            (model.max_position_embeddings, model.head_dim if model.rotary_dim is None else model.rotary_dim),
            model.compute_dtype, persistent=True)
        # Exact-class OOT registration selects this constructor only for the
        # default RotaryEmbedding. Llama3/native and scaled constructors have
        # different persistent programs; do not infer their storage here.
        self._rope_constructor_cache_views = (_ascend018_interleaved_rope_constructor_views(trace, self.rope_cache)
            if ascend018 and model.rope_scaling_type == "default" else ())
        self.layers = []
        for layer in range(*layer_range):
            prefix = f"layer-{layer}"
            cache = None if profile_no_metadata else trace.tensor(prefix + ":kv-cache",
                gqa_cache_shape(cache_capacity_tokens, model.kv_heads_per_tensor_parallel_rank(tp_size), model.head_dim),
                model.compute_dtype, persistent=True)
            self.layers.append(self.decoder_type(trace, model, pool, prefix,
                self.rope_cache, cache, q_lens, kv_lens, slot_mapping,
                profile_no_metadata=profile_no_metadata, group=group, collective=collective,
                mlp_factory=None if mlp_factories is None else mlp_factories[layer], rope_program=rope_program,
                hardware_id=hardware_id, ffn_factory=ffn_factory, collective_inplace=ascend018,
                model_execution_version=self.model_execution_version))
        self.norm = _dense_final_norm(trace, model, pool, hardware_id) if self.last_stage else None

    def __call__(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        # Qwen2Model.forward's ordinary PP/no-aux path, inherited by Qwen3.
        if self.first_stage:
            if intermediate_tensors is not None:
                raise ValueError("first pipeline stage cannot consume intermediate tensors")
            if inputs_embeds is not None:
                self.trace._check_tensor(inputs_embeds)
                if (inputs_embeds.shape != self.inputs_embeds_shape
                        or inputs_embeds.dtype != self.inputs_embeds_dtype):
                    raise ValueError("model inputs_embeds requires the declared typed hidden-state extent")
                hidden_states = inputs_embeds
            else:
                if self.embed_tokens is None:
                    raise ValueError("model constructed for inputs_embeds cannot consume token IDs")
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            if inputs_embeds is not None:
                raise ValueError("inputs_embeds belongs to the first model stage")
            if not isinstance(intermediate_tensors, dict) or set(intermediate_tensors) != {"hidden_states", "residual"}:
                raise ValueError("later pipeline stages require both intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers:
            if self._source_module_program is None:
                hidden_states, residual = layer(positions, hidden_states, residual)
            else:
                hidden_states, residual = self.trace._call_module_frame(self._source_module_program,
                    layer._source_prefix, layer, (positions, hidden_states, residual))
        if not self.last_stage:
            return {"hidden_states": hidden_states, "residual": residual}
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaModel(Qwen3Model):
    decoder_type = LlamaDecoderLayer


class Glm4MoeAttention(Qwen3Attention):
    """GLM4 standard GQA body, including optional QK norm and partial RoPE."""

    def __init__(self, trace, model, *args, **kwargs):
        self.qk_norm = model.qk_norm
        super().__init__(trace, model, *args, **kwargs)

    def __call__(self, positions, hidden_states):
        if self.qk_norm:
            return Qwen3Attention.__call__(self, positions, hidden_states)
        return LlamaAttention.__call__(self, positions, hidden_states)


class Glm4MoE:
    """vLLM0.11 Glm4MoE.forward; separate shared branch, then late TP sum."""

    def __init__(self, trace, model, pool, prefix, *, resolution, invocation_id, group=None, collective=None):
        if group is not None or collective is not None:
            raise NotImplementedError("GLM4 common forward currently requires its maintained TP1 domain")
        self.cast = SymbolicFloat32Cast(trace, prefix + ":router-cast", pool)
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size,
            model.num_experts, kind=WorkloadKind.ROUTER)
        self.router = SymbolicBiasedSigmoidTopK(trace, prefix + ":route", pool, model)
        self.experts = SymbolicUnquantizedExperts(trace, prefix + ":experts", pool, model,
            tp_size=1, resolution=resolution, invocation_id=invocation_id, inplace=True, model_execution_version="vllm_0_11_0")
        self.shared = (DeepseekV2MLP(trace, model, pool, prefix + ":shared", tp_size=1,
            intermediate_size=model.moe_intermediate_size * model.num_shared_experts,
            module_path=prefix + ".shared_experts", reduce_results=False)
            if model.num_shared_experts else None)
        self.scale = SymbolicScalarMultiply(trace, prefix + ":output-scale", pool,
            model.routed_scaling_factor, inplace=False)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)

    def __call__(self, hidden_states):
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view_as_shape(-1, hidden_dim)
        router_logits = self.gate(self.cast(hidden_states))[0]
        # FusedMoE.forward_impl's non-modular branch computes shared first;
        # UnquantizedFusedMoEMethod then calls fused_experts(inplace=True).
        shared_output = None if self.shared is None else self.shared(hidden_states)
        weights, ids = self.router(hidden_states, router_logits)
        output = self.experts(hidden_states, weights, ids)
        output = self.scale(output)
        if shared_output is not None:
            output = self.add(output, shared_output)
        return output.view_as_shape(num_tokens, hidden_dim)


class Glm4MoeDecoderLayer(Qwen3DecoderLayer):
    attention_type = Glm4MoeAttention


class Glm4MoeModel(Qwen3Model):
    """Glm4MoeModel uses the same embedding/residual loop and final norm."""
    decoder_type = Glm4MoeDecoderLayer
    model_execution_version = "vllm_0_11_0"


class Glm4MoeMultiTokenPredictorLayer:
    """vLLM0.11 draft-config layer0 body; weights/KV reused across passes.

    speculative.py resets num_hidden_layers=0 before constructing GLM4's
    draft. The maintained first_k_dense_replace=1 source therefore has a
    dense decoder, distinct from the0.30 appended-layer MoE constructor.
    The caller supplies genuine proposer metadata; this class neither
    invents next-token values nor performs sampling/rejection decisions.
    """

    def __init__(self, trace, model, pool, prefix, rope_cache, cache, q_lens, kv_lens, slots, *, ffn_factory=None):
        _validate_glm4_source(model)
        if (model.mtp_architecture != "glm4_moe_mtp" or model.num_mtp_layers != 1
                or model.mtp_layer_uses_moe(0)):
            raise NotImplementedError("GLM4 draft source requires its single dense0.11 predictor layer")
        self.trace, self.model = trace, model
        self.embed_tokens = SymbolicEmbedding(trace, prefix + ":embedding", pool, model.hidden_size, "bf16",
            model_execution_version="vllm_0_11_0")
        self.mask = SymbolicPositionZeroMask(trace, prefix + ":position-mask", pool)
        self.enorm = SymbolicRMSNorm(trace, prefix + ":embedding-norm", pool, model.hidden_size, model.rms_norm_eps)
        self.hnorm = SymbolicRMSNorm(trace, prefix + ":hidden-norm", pool, model.hidden_size, model.rms_norm_eps)
        self.cat = SymbolicConcat(trace, prefix + ":concat", pool)
        self.eh_proj = SymbolicLinear(trace, prefix + ":merge", pool, 2 * model.hidden_size, model.hidden_size)
        self.mtp_block = Glm4MoeDecoderLayer(trace, model, pool, prefix + ":decoder", rope_cache,
            cache, q_lens, kv_lens, slots, hardware_id="nvidia_a800", ffn_factory=ffn_factory)
        self.add = SymbolicAdd(trace, prefix + ":residual-add", pool)
        self.head_norm = SymbolicRMSNorm(trace, prefix + ":head-norm", pool, model.hidden_size, model.rms_norm_eps)
        self.head = SymbolicLinear(trace, prefix + ":head", pool, model.hidden_size, model.vocab_size,
            kind=WorkloadKind.LM_HEAD)

    def __call__(self, input_ids, positions, previous_hidden_states, *, position_values,
                 q_lens, kv_lens, slots, inputs_embeds=None, slot_tensor=None):
        rows = sum(q_lens)
        if (not isinstance(q_lens, tuple) or not q_lens or not isinstance(kv_lens, tuple) or len(q_lens) != len(kv_lens)
                or any(type(q) is not int or type(k) is not int or not 0 < q <= k
                    for q, k in zip(q_lens, kv_lens))
                or input_ids.shape != (rows,) or input_ids.dtype not in ("int32", "int64")
                or previous_hidden_states.shape != (rows, self.model.hidden_size)
                or previous_hidden_states.dtype != "bf16" or len(slots) != rows
                or len(set(slots)) != rows
                or not isinstance(position_values, tuple) or len(position_values) != rows
                or any(type(p) is not int or not 0 <= p < self.model.max_position_embeddings for p in position_values)):
            raise ValueError("GLM4 MTP requires its declared per-pass query/KV/input/hidden metadata")
        attention = self.mtp_block.self_attn.attn
        if (sum(kv_lens) > attention.cache.shape[1]
                or any(type(slot) is not int or not 0 <= slot < attention.cache.shape[1] for slot in slots)):
            raise ValueError("GLM4 MTP slots must be unique valid positions in its cache")
        attention.q_lens, attention.kv_lens, attention.slot_mapping = q_lens, kv_lens, slots
        attention.slot_tensor = slot_tensor
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.mask(positions, inputs_embeds, position_values=position_values)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(self.cat(inputs_embeds, previous_hidden_states))[0]
        hidden_states, residual = self.mtp_block(positions, hidden_states, None)
        hidden_states = self.add(residual, hidden_states)
        return hidden_states

    def compute_logits(self, hidden_states):
        return self.head(self.head_norm(hidden_states))[0]


def _validate_glm4_source(model):
    if (model.model_type != "glm4_moe" or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
            or model.hidden_act != "silu" or model.rms_norm_eps is None or model.rotary_dim is None
            or model.rope_scaling_type != "default" or model.attention_bias or model.mlp_bias
            or model.moe_group_count != 1 or model.moe_topk_group_count != 1):
        raise NotImplementedError("GLM4 common source requires BF16, explicit norms/partial RoPE and single-group routing")


def _glm4_model_inputs(model, step, pool, expert_routing, *, input_ids, positions,
        slot_mapping, cache_capacity_tokens, source_profile, mtp_source=False,
        trace=None, ffn_factory=None, expert_pool=None):
    """Shared target model construction and typed inputs, before forward."""
    from functools import partial
    _validate_glm4_source(model)
    config = pool.parallel_plan.config
    if (pool.hardware_id not in ("nvidia_a800", "nvidia_h200")
            or mtp_source and pool.hardware_id != "nvidia_a800" or config.enable_expert_parallel
            or any(value != 1 for value in (config.tensor_parallel_size, config.pipeline_parallel_size,
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("GLM4 common decoder requires A800/H200 TP1; MTP retains its A800 source")
    q_lens, kv_lens, rows = _dense_source_metadata(model, step, input_ids=input_ids,
        positions=positions, slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, hardware_id=pool.hardware_id, mtp_source=mtp_source)
    trace = ForwardTrace(rank=pool.ranks[0], namespace=step.step_id) if trace is None else trace
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool if expert_pool is None else expert_pool,
        expert_routing, invocations, rows)
    factories = tuple(partial(Glm4MoE, resolution=resolutions[layer], invocation_id=invocations[layer])
        if model.layer_uses_moe(layer) else None for layer in range(model.num_hidden_layers))
    network = Glm4MoeModel(trace, model, pool.name, q_lens, kv_lens, cache_capacity_tokens, slot_mapping,
        mlp_factories=factories, hardware_id=pool.hardware_id, ffn_factory=ffn_factory)
    ids = trace.tensor("input_ids", (rows,), "int32")
    pos = trace.tensor("positions", (rows,), "int64")
    return trace, network, ids, pos, q_lens, kv_lens


def trace_glm4_decoder(model, step, pool, expert_routing, **source_inputs):
    """Complete target hidden-state forward; caller owns sampling/proposing."""
    trace, network, ids, pos, _, _ = _glm4_model_inputs(model, step, pool, expert_routing,
        mtp_source=False, **source_inputs)
    hidden_states = network(ids, pos)
    return trace, hidden_states


def trace_glm4_forward(model, step, pool, expert_routing, *,
        target_sampling_source="vllm_0_11_plain_greedy_bf16", **source_inputs):
    """Target-off GLM4 with explicit logits, FP32 cast or greedy output."""
    if target_sampling_source not in (None, "vllm_0_11_plain_sampler_bf16", "vllm_0_11_plain_greedy_bf16"):
        raise ValueError("GLM4 target output requires its matching0.11 sampler source")
    trace, hidden_states = trace_glm4_decoder(model, step, pool, expert_routing, **source_inputs)
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    indices = tuple(sum(q_lens[:i + 1]) - 1 for i in range(len(q_lens)))
    logits_indices = trace.tensor("logits_indices", (len(indices),), "int32")
    selected = SymbolicLogitsRowSelect(trace, "logits-select", pool.name)(hidden_states, logits_indices,
        selected_indices=indices, model_execution_version="vllm_0_11_0", hardware_id=pool.hardware_id)
    logits, _ = SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
        model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected)
    if target_sampling_source is not None:
        SymbolicGreedy(trace, "target-sampler", pool.name)(logits, model_execution_version="vllm_0_11_0",
            sample_logits=target_sampling_source == "vllm_0_11_plain_greedy_bf16")
    return trace


def trace_glm4_mtp_forward(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
        cache_capacity_tokens, source_profile, max_model_len, max_batched_tokens, max_requests,
        trace=None, ffn_factory=None, expert_pool=None):
    """Complete0.11 eager all-greedy padded target/rejection/Eagle-MTP graph.

    Acceptance and positions are explicit schedule scenarios; no token is
    computed. Logical KV coordinates are disjoint request arenas, not native
    physical pages. This initial source domain rejects context-cap overflow,
    discarded/incomplete rows, trees, auxiliary hidden states and parallelism.
    """
    from .traced_sampling import MTPSourceOps
    requests = step.requests
    if (any(type(cap) is not int or cap <= 0 for cap in (max_model_len, max_batched_tokens, max_requests))
            or max_model_len > model.max_position_embeddings or len(requests) > max_requests
            or sum(r.scheduled_tokens for r in requests) > max_batched_tokens):
        raise ValueError("GLM4 MTP requires explicit source capacities covering every active row/request")
    depths = {r.mtp_proposer_tokens for r in requests}
    if (len(depths) != 1 or not depths or min(depths) <= 0
            or any(r.mtp_proposer_mode != "padded" or r.raw_model_output_tokens != r.mtp_accepted_tokens + 1
                or r.scheduled_tokens < r.mtp_verified_tokens + 1 for r in requests)):
        raise NotImplementedError("GLM4 MTP source requires one padded depth and a valid sampled anchor for every request")
    depth = next(iter(depths))
    if pool.kv_cache is None:
        raise ValueError("GLM4 MTP requires an explicit cache block-size contract")
    # Eagle samples the first draft at the retained target row, then only
    # advances that position for the remaining passes. Rejected target-tail
    # rows stay in the padded first pass but do not advance that anchor.
    if any(r.computed_tokens + r.scheduled_tokens > max_model_len
            or r.computed_tokens + r.scheduled_tokens - r.mtp_verified_tokens
                + r.mtp_accepted_tokens + depth - 2 >= max_model_len for r in requests):
        raise NotImplementedError("GLM4 MTP cap-exceeded padding/cache slots are not migrated")
    trace, network, ids, pos, q_lens, kv_lens = _glm4_model_inputs(model, step, pool, expert_routing,
        input_ids=input_ids, positions=positions, slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, mtp_source=True, trace=trace, ffn_factory=ffn_factory, expert_pool=expert_pool)
    ops = MTPSourceOps(trace, "mtp-source", pool.name)
    rows, batch = sum(q_lens), len(q_lens)
    verified, accepted = tuple(r.mtp_verified_tokens for r in requests), tuple(r.mtp_accepted_tokens for r in requests)
    source_has_spec = any(verified)

    # EagleProposer constructor buffers, including the unused pure-text
    # inputs_embeds arena. Capacities are explicit; active slices are views.
    draft_ids = trace.tensor("mtp.input-ids", (max_batched_tokens,), "int32", persistent=True)
    draft_pos = trace.tensor("mtp.positions", (max_batched_tokens,), "int64", persistent=True)
    draft_hidden = trace.tensor("mtp.hidden-states", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
    unused_embeds = trace.tensor("mtp.inputs-embeds", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
    arange = trace.tensor("mtp.arange", (max(max_requests + 1, max_batched_tokens),), "int32", persistent=True)
    backup = trace.tensor("mtp.backup-next-token-ids", (max_requests,), "int32", persistent=True)
    query_start = trace.tensor("parent.query-start", (max_requests + 1,), "int32", persistent=True)
    seq_lens = trace.tensor("parent.seq-lens", (max_requests,), "int32", persistent=True).narrow(0, 0, batch)
    # This is only the source table's logical active-read descriptor. Its
    # physical backing/IDs continue to belong to the native cache owner.
    block_size = next(tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm")
    block_table = trace.tensor("parent.block-table", (max_requests, -(-max_model_len // block_size)),
        "int32", persistent=True).narrow(0, 0, batch)
    parent_slots = trace.tensor("parent.slots", (rows,), "int64", persistent=True)
    arena_bases, arena_size = [], 0
    for length in kv_lens:
        arena_bases.append(arena_size)
        arena_size += -(-(length + depth - 1) // block_size) * block_size
    draft_slots = tuple(arena_bases[i] + r.computed_tokens + j for i, r in enumerate(requests)
        for j in range(r.scheduled_tokens))
    if slot_mapping != draft_slots:
        raise ValueError("GLM4 MTP target and draft must share the declared logical page coordinates")
    draft_cache = trace.tensor("mtp.kv-cache", (2, arena_size, model.num_key_value_heads, model.head_dim),
        "bf16", persistent=True)
    predictor = Glm4MoeMultiTokenPredictorLayer(trace, model, pool.name, "mtp", network.rope_cache,
        draft_cache, q_lens, kv_lens, draft_slots, ffn_factory=ffn_factory)

    # GPUModelRunner._calc_spec_decode_metadata: all four index/prefix
    # arrays are INT32 (explicit cumsum_dtype=np.int32 in0.11).
    logits_indices_values, bonus_values, target_values = [], [], []
    offset, sampled_offset = 0, 0
    for query, count in zip(q_lens, verified):
        logits_indices_values.extend(range(offset + query - count - 1, offset + query))
        bonus_values.append(sampled_offset + count)
        target_values.extend(range(sampled_offset, sampled_offset + count))
        offset += query
        sampled_offset += count + 1
    logits_indices_values, bonus_values, target_values = map(tuple,
        (logits_indices_values, bonus_values, target_values))
    logits_indices = trace.tensor("target.logits-indices", (len(logits_indices_values),), "int32")
    if source_has_spec:
        ops.h2d(logits_indices)
        prefix = ops.h2d(trace.tensor("target.cu-draft-tokens", (batch,), "int32"))
        bonus_indices = ops.h2d(trace.tensor("target.bonus-indices", (batch,), "int32"))
        target_indices = ops.h2d(trace.tensor("target.draft-indices", (sum(verified),), "int32"))
        selected_ids = ops.gather(ids, logits_indices, selected_indices=logits_indices_values)
        shifted = ops.unary(target_indices, expression="add", scalar=1)
        previous_draft_ids = ops.gather(selected_ids, shifted, selected_indices=tuple(i + 1 for i in target_values))
    hidden_states = network(ids, pos)
    selected_hidden = ops.gather(hidden_states, logits_indices, selected_indices=logits_indices_values)
    logits = SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
        model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected_hidden)[0]
    def sample_target():
        if source_has_spec:
            bonus_logits = ops.gather(logits, bonus_indices, selected_indices=bonus_values)
            bonus_ids = SymbolicGreedy(trace, "target-bonus-sampler", pool.name)(bonus_logits,
                model_execution_version="vllm_0_11_0").view_as_shape(batch)
            target_logits = ops.gather(logits, target_indices, selected_indices=target_values)
            #0.11 compute_probs(all_greedy=True) returns the original BF16 logits.
            target_argmax = ops.unary(target_logits, "argmax")
            sampled = ops.fill(trace.tensor("target.rejection-output", (batch, max(verified) + 1), "int32"), value=-1)
            return ops.rejection(sampled, prefix, previous_draft_ids, target_argmax, bonus_ids,
                verified=verified, accepted=accepted)
        else:
            return SymbolicGreedy(trace, "target-sampler", pool.name)(logits, model_execution_version="vllm_0_11_0")
    sampled = sample_target()

    # prepare_next_token_ids_padded: actual Tensor expressions, with INT64
    # bool.sum output. No discarded requests in this declared source domain.
    def prepare_next():
        backup_front = ops.h2d(backup.narrow(0, 0, batch))
        valid_sampled = ops.copy(sampled)
        if sampled.shape[1] == 1:
            valid_mask = ops.fill(trace.tensor("mtp.valid-mask", sampled.shape, "bool"), value=True)
        else:
            valid_mask = ops.binary(ops.unary(valid_sampled, "compare", expression="not_equal", scalar=-1),
                ops.unary(valid_sampled, "compare", expression="less", scalar=model.vocab_size), "logical_and")
        valid_count = ops.unary(valid_mask, "sum")
        last_valid = ops.unary(valid_count, expression="subtract", scalar=1)
        safe_last = ops.unary(last_valid, expression="clamp_min", scalar=0)
        selected_tokens = ops.row_gather(valid_sampled, safe_last)
        next_tokens = ops.where(ops.unary(last_valid, "compare", expression="not_equal", scalar=-1), selected_tokens, backup_front)
        return next_tokens, valid_count
    next_tokens, valid_count = prepare_next()
    def prepare_source_inputs():
        if source_has_spec:
            # torch.cat([cu[0:1], cu[1:]-cu[:-1]]) also materializes for B=1.
            first = prefix.narrow(0, 0, 1)
            if batch > 1:
                difference = ops.binary(prefix.narrow(0, 1, batch - 1), prefix.narrow(0, 0, batch - 1))
                num_draft = ops.concat((first, difference))
            else:
                num_draft = ops.concat((first,))
            positive = ops.unary(num_draft, "compare", expression="greater", scalar=0)
            plus_one = ops.unary(num_draft, expression="add", scalar=1)
            rejected_candidate = ops.binary(plus_one, valid_count, "promote_subtract")
            zero_draft = ops.fill(trace.tensor("mtp.zero-draft-count", (batch,), "int32"))
            rejected = ops.where(positive, rejected_candidate, zero_draft)
            source_metadata = {"slot_mapping": ops.gather(parent_slots, arange.narrow(0, 0, rows),
                selected_indices=tuple(range(rows)))}
            query_end = ops.unary(query_start.narrow(0, 1, batch), expression="subtract", scalar=1)
            last_indices = ops.binary(query_end, rejected, "promote_subtract")
            retained = tuple(sum(q_lens[:i]) + q_lens[i] - (verified[i] - accepted[i]) - 1 for i in range(batch))
            # The three advanced index expressions copy complete padded inputs,
            # even though arange selects every row in order.
            target_ids = ops.gather(ids, arange.narrow(0, 0, rows), selected_indices=tuple(range(rows)))
            target_pos = ops.gather(pos, arange.narrow(0, 0, rows), selected_indices=tuple(range(rows)))
            target_hidden = ops.gather(hidden_states, arange.narrow(0, 0, rows), selected_indices=tuple(range(rows)))
        else:
            retained = tuple(sum(q_lens[:i + 1]) - 1 for i in range(batch))
            last_indices = ops.unary(query_start.narrow(0, 1, batch), expression="subtract", scalar=1)
            target_ids, target_pos, target_hidden = ids, pos, hidden_states
            source_metadata = {"slot_mapping": parent_slots}
        return target_ids, target_pos, target_hidden, last_indices, retained, source_metadata
    target_ids, target_pos, target_hidden, last_indices, retained, source_metadata = prepare_source_inputs()
    id_front, pos_front = draft_ids.narrow(0, 0, rows), draft_pos.narrow(0, 0, rows)
    hidden_front = draft_hidden.narrow(0, 0, rows)
    if rows > 1:
        ops.copy(target_ids.narrow(0, 1, rows - 1), id_front.narrow(0, 0, rows - 1))
    ops.scatter(next_tokens, last_indices, id_front, selected_indices=retained)
    ops.copy(target_pos, pos_front)
    ops.copy(target_hidden, hidden_front)
    ret_hidden = predictor(id_front, pos_front, hidden_front, position_values=positions,
        q_lens=q_lens, kv_lens=kv_lens, slots=draft_slots, slot_tensor=source_metadata["slot_mapping"])
    sample_hidden = ops.gather(ret_hidden, last_indices, selected_indices=retained)
    draft_logits = predictor.compute_logits(sample_hidden)
    if depth == 1:
        output = ops.unary(draft_logits, "argmax").view_as_shape(batch, 1)
        return trace
    selected_positions = ops.gather(target_pos, last_indices, selected_indices=retained)
    recurrent_hidden = ops.gather(ret_hidden, last_indices, selected_indices=retained)
    generated = [ops.unary(draft_logits, "argmax")]
    current_positions = tuple(positions[index] for index in retained)
    for draft_index in range(1, depth):
        next_ids = ops.unary(generated[-1], "cast_ids")
        ops.unary(selected_positions, inplace=True, expression="add", scalar=1)
        exceeds = ops.unary(selected_positions, "compare", expression="greater_equal", scalar=max_model_len)
        clamped_positions = ops.where(exceeds, selected_positions)
        ops.unary(seq_lens, inplace=True, expression="add", scalar=1)
        ops.masked_fill(seq_lens, exceeds, masked_rows=0)
        # Selected block-ID gather/address arithmetic; table reads are
        # represented by the selected INT32 port, not invented page values.
        block_numbers = ops.unary(clamped_positions, expression="floor_divide", scalar=block_size)
        gathered_blocks = ops.row_gather(block_table, block_numbers)
        scaled_blocks = ops.unary(gathered_blocks, expression="multiply", scalar=block_size)
        remainders = ops.unary(clamped_positions, expression="remainder", scalar=block_size)
        slot_values = ops.binary(scaled_blocks, remainders, "promote_add")
        source_metadata["slot_mapping"] = ops.masked_fill(slot_values, exceeds, masked_rows=0)
        ops.copy(next_ids, draft_ids.narrow(0, 0, batch))
        ops.copy(clamped_positions, draft_pos.narrow(0, 0, batch))
        ops.copy(recurrent_hidden, draft_hidden.narrow(0, 0, batch))
        current_positions = tuple(p + 1 for p in current_positions)
        current_slots = tuple(arena_bases[i] + p for i, p in enumerate(current_positions))
        ret_hidden = predictor(draft_ids.narrow(0, 0, batch), draft_pos.narrow(0, 0, batch),
            draft_hidden.narrow(0, 0, batch), position_values=current_positions,
            q_lens=(1,) * batch, kv_lens=tuple(k + draft_index for k in kv_lens), slots=current_slots,
            slot_tensor=source_metadata["slot_mapping"])
        recurrent_hidden = ret_hidden.narrow(0, 0, batch)
        draft_logits = predictor.compute_logits(ret_hidden.narrow(0, 0, batch))
        generated.append(ops.unary(draft_logits, "argmax"))
    output = ops.stack(tuple(generated))
    return trace


class Qwen3MoeRunner:
    """Ordinary unquantized DP/SP1 Runner with a separately owned gate."""

    def __init__(self, trace, model, pool, prefix, *, group, collective, resolution, invocation_id, ep_size=1,
                 model_execution_version="vllm_0_30_0"):
        self.trace, self.group, self.collective, self.pool, self.prefix = trace, group, collective, pool, prefix
        self.ep_size, self.invocation_id = ep_size, invocation_id
        self.collective_inplace = model_execution_version == "vllm_0_18_0"
        self.owner_mask = (SymbolicExpertOwnerMask(trace, prefix + ":owner-mask", pool, model,
            expert_rank=group.participants.index(trace.rank), expert_parallel_size=ep_size)
            if self.collective_inplace else None)
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size,
            model.num_experts, kind=WorkloadKind.ROUTER)
        self.router = SymbolicSoftmaxTopK(trace, prefix + ":route", pool, model,
            model_execution_version=model_execution_version)
        self.experts = SymbolicUnquantizedExperts(trace, prefix + ":experts", pool, model,
            tp_size=1 if group is None else len(group.traces), resolution=resolution, invocation_id=invocation_id,
            expert_parallel_size=ep_size, expert_rank=group.participants.index(trace.rank) if ep_size > 1 else 0,
            inplace=model_execution_version == "vllm_0_11_0",
            routing_weight_dtype="bf16" if self.collective_inplace else "fp32", model_execution_version=model_execution_version)

    def __call__(self, *, hidden_states, router_logits):
        if hidden_states is not router_logits:
            raise ValueError("Qwen3 Runner must own the gate over the source hidden-state placeholder")
        logits, _ = self.gate(hidden_states)
        weights, ids = self.router(hidden_states, logits)
        if self.owner_mask is not None:
            weights = self.owner_mask(weights, ids)
        output = self.experts(hidden_states, weights, ids)
        if self.group is not None:
            return self.group.call(self.prefix + ":all-reduce", output,
                pool_name=self.pool, config=self.collective, inplace=self.collective_inplace,
                moe_invocation_id=self.invocation_id if self.ep_size > 1 else None)
        return output


class Qwen3MoeSparseMoeBlock:
    def __init__(self, trace, model, pool, prefix, *, group, collective, resolution, invocation_id, ep_size=1,
                 model_execution_version="vllm_0_30_0", collective_inplace=False):
        if collective_inplace != (model_execution_version == "vllm_0_18_0"):
            raise ValueError("MoE collective storage must match its model source")
        self.is_sequence_parallel = False
        self.experts = Qwen3MoeRunner(trace, model, pool, prefix,
            group=group, collective=collective, resolution=resolution, invocation_id=invocation_id, ep_size=ep_size,
            model_execution_version=model_execution_version)

    def __call__(self, hidden_states):
        assert hidden_states.dim() <= 2, (
            "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view_as_shape(-1, hidden_dim)
        if self.is_sequence_parallel:
            raise NotImplementedError("Qwen3 virtual MoE sequence parallelism is not migrated")
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


def _validate_qwen3_moe_source(model, config):
    if (model.model_type != "qwen3_moe" or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
            or model.hidden_act != "silu" or model.rope_scaling_type != "default"
            or model.num_shared_experts != 0 or type(model.moe_normalize_topk_probability) is not bool
            or (config.enable_expert_parallel and (config.expert_parallel_size != 2 or config.tensor_parallel_size != 2))
            or any(value != 1 for value in (
                config.pipeline_parallel_size, config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("virtual Qwen3 MoE requires BF16, default RoPE, PP/SP/CP/DP1 and EP1 or TP=EP=2")


def _vllm011_qwen3_moe_ep2_source(model, pool):
    return _versioned_qwen3_moe_ep2_source(model, pool, "vllm_0_11_0")


def _versioned_qwen3_moe_ep2_source(model, pool, model_execution_version):
    hardware = {"vllm_0_11_0": ("nvidia_a800", "nvidia_h200"), "vllm_0_24_0": ("metax_c500",), "vllm_0_18_0": ("ascend_910b",)}
    config = pool.parallel_plan.config
    return (model_execution_version in hardware and pool.hardware_id in hardware[model_execution_version]
        and is_validated_qwen3_moe_ep2_model_scope(model)
        and model.num_experts in (2, 128) and config.tensor_parallel_size == 2
        and config.enable_expert_parallel and config.expert_parallel_size == 2
        and len(pool.ranks) == 2 and config.is_rank_local
        and all(value == 1 for value in (config.pipeline_parallel_size, config.sequence_parallel_size,
            config.context_parallel_size, config.data_parallel_size)))


def trace_qwen3_moe_forward(model, step, pool, expert_routing, *, input_ids, positions,
        slot_mapping, cache_capacity_tokens, source_profile, model_execution_version="vllm_0_30_0", sample_logits=True,
        cast_logits=False, graph_token_shape=None, graph_eager_kinds=(), source_namespace=None):
    """Full Qwen3-MoE stack on the same GQA/residual/target/storage primitives."""
    from functools import partial
    config = pool.parallel_plan.config
    _validate_qwen3_moe_source(model, config)
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
        raise NotImplementedError("Qwen3 MoE requires its version-bound source")
    if model_execution_version == "vllm_0_11_0" and (
            pool.hardware_id not in ("nvidia_a800", "nvidia_h200")
            or (pool.hardware_id == "nvidia_h200" and not _vllm011_qwen3_moe_ep2_source(model, pool))
            or (not _vllm011_qwen3_moe_ep2_source(model, pool) and (config.tensor_parallel_size != 1
                or len(pool.ranks) != 1 or config.enable_expert_parallel))
            or source_profile != "cuda_eager_inplace"
            or expert_routing.source_kind not in ("explicit_assumption", "profile_observation")
            #0.11 uses a positive-byte softmax workspace outside this fused
            # power-of-two domain; that storage is not silently omitted.
            or model.num_experts > 256 or model.num_experts & (model.num_experts - 1)):
        raise NotImplementedError("vLLM0.11 Qwen3 MoE requires A800 TP1/EP-off or A800/H200 validated TP=EP2, explicit seeded or profiled routing and power-of-two experts <=256")
    if model_execution_version in ("vllm_0_18_0", "vllm_0_24_0") and not _versioned_qwen3_moe_ep2_source(model, pool, model_execution_version):
        raise NotImplementedError("C5000.24 MoE requires its validated TP=EP2 ordinary source")
    if graph_token_shape is not None and (model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0")
            or not _versioned_qwen3_moe_ep2_source(model, pool, model_execution_version) or sample_logits is not True):
        raise NotImplementedError("MoE capture requires its declared versioned TP=EP2 target source")
    rows = (sum(request.scheduled_tokens for request in step.requests) if graph_token_shape is None
        else graph_token_shape.model_input_tokens)
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
    # Routing always resolves against the actual ScheduleStep and invocation
    # identity above. Only source object names may use a stable program key.
    if source_namespace is not None and (type(source_namespace) is not str or not source_namespace):
        raise ValueError("source namespace must be an explicit nonempty string")
    source_step = step if source_namespace is None else replace(step, step_id=source_namespace)
    factories = tuple(partial(Qwen3MoeSparseMoeBlock, resolution=resolution, invocation_id=invocation,
        ep_size=config.expert_parallel_size, model_execution_version=model_execution_version)
        for resolution, invocation in zip(resolutions, invocations, strict=True))
    source_options = dict(hardware_id=pool.hardware_id,
        fused_embedding=(pool.hardware_id not in ("metax_c500", "ascend_910b")
            and not (model_execution_version == "vllm_0_11_0" and config.tensor_parallel_size > 1)),
        rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type),
        model_execution_version=model_execution_version, graph_token_shape=graph_token_shape,
        graph_eager_kinds=graph_eager_kinds)
    if config.tensor_parallel_size == 1:
        return _trace_dense_forward(model, source_step, input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_profile=source_profile, rank=pool.ranks[0],
            namespace=source_step.step_id, model_type=Qwen3Model, mlp_factories=factories,
            sample_logits=sample_logits, cast_logits=cast_logits, **source_options)
    if sample_logits is not True and not (pool.hardware_id == "nvidia_h200"
            and model_execution_version == "vllm_0_30_0" and config.tensor_parallel_size == 2
            and (not config.enable_expert_parallel or config.expert_parallel_size == 2
                and is_validated_qwen3_moe_ep2_model_scope(model))):
        raise NotImplementedError("Qwen3 MoE static LM-head-only source requires TP1 or existing H200 TP2/validated TP=EP2 source")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{source_step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=source_step.step_id)
    for trace in traces:
        _trace_dense_forward(model, source_step, input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_profile=source_profile, rank=trace.rank,
            namespace=trace.namespace, model_type=Qwen3Model, trace=trace, group=group,
            collective=pool.collective_for("all_reduce"), gather=pool.collective_for("all_gather"),
            mlp_factories=factories, sample_logits=sample_logits, cast_logits=cast_logits, **source_options)
    if graph_token_shape is not None:
        _remap_collective_capture_nodes(group)
    return group


class KimiK3MLP:
    """Ordinary TP1 KimiMLP body with the source SiTU primitive."""

    def __init__(self, trace, model, pool, prefix, *, shared=False):
        width = model.moe_intermediate_size * model.num_shared_experts if shared else model.intermediate_size
        self.gate_up_proj = SymbolicLinear(trace, prefix + ":gate-up", pool, model.hidden_size, 2 * width)
        self.act_fn = SymbolicK3SiTU(trace, prefix + ":situ", pool, model)
        self.down_proj = SymbolicLinear(trace, prefix + ":down", pool, width, model.hidden_size)

    def __call__(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class KimiK3MoE:
    """TP1 modular Marlin: latent routed branch and original-input shared branch."""

    def __init__(self, trace, model, pool, prefix, *, resolution, invocation_id, layer_index):
        latent = model.kimi_k3_semantics.routed_expert_hidden_size
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size,
            model.num_experts, kind=WorkloadKind.ROUTER, output_dtype="fp32")
        self.down = SymbolicLinear(trace, prefix + ":latent-down", pool, model.hidden_size, latent)
        self.router = SymbolicK3Routing(trace, prefix + ":route", pool, model)
        self.shared = KimiK3MLP(trace, model, pool, prefix + ":shared", shared=True)
        self.experts = SymbolicK3Experts(trace, prefix + ":experts", pool, model,
            resolution=resolution, invocation_id=invocation_id, layer_index=layer_index)
        self.norm = SymbolicRMSNorm(trace, prefix + ":latent-norm", pool, latent, model.kimi_k3_semantics.rms_norm_eps)
        self.up = SymbolicLinear(trace, prefix + ":latent-up", pool, latent, model.hidden_size)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)

    def _runner(self, hidden_states, router_logits, shared_experts_input):
        # TP1 LatentMoERunner uses the ordinary output transform after expert
        # combination; no fused TP reduction or MegaMoE beta-add is selected.
        shared_output = self.shared(shared_experts_input)
        weights, ids = self.router(router_logits)
        fused_output = self.experts(hidden_states, weights, ids)
        fused_output = self.norm(fused_output)
        fused_output, _ = self.up(fused_output)
        return self.add(shared_output, fused_output)

    def __call__(self, hidden_states):
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view_as_shape(-1, hidden_size)
        router_output, _ = self.gate(hidden_states)
        routed_hidden_states, _ = self.down(hidden_states)
        final_hidden_states = self._runner(routed_hidden_states, router_output, hidden_states)
        return final_hidden_states.view_as_shape(num_tokens, hidden_size)


class KimiK3KDADecode:
    """Eager ordinary Triton KDA wrapper, without executing its tensor values."""

    def __init__(self, trace, model, pool, prefix, *, rows, state_indices, state_pool_slots):
        from .shape_inference import infer_gemm_dimensions
        self.trace, self.prefix, self.pool = trace, prefix, pool
        source = model.kimi_k3_semantics
        self.heads, self.dim = source.kda_num_heads, source.kda_head_dim
        self.width, self.rows = self.heads * self.dim, rows
        _, packed, hidden = infer_gemm_dimensions(model, "in_proj_qkvgfab", rows, tp_size=1)
        self.padding = packed - (4 * self.width + self.dim + self.heads)
        self.in_proj_qkvgfab = SymbolicLinear(trace, prefix + ":packed", pool, hidden, packed)
        self.f_b_proj = SymbolicLinear(trace, prefix + ":f-b", pool, self.dim, self.width)
        self.o_proj = SymbolicLinear(trace, prefix + ":out", pool, self.width, hidden)
        self.conv_state = trace.tensor(prefix + ":conv-state",
            (state_pool_slots, 3 * self.width, source.short_conv_kernel_size - 1), "bf16", persistent=True)
        self.recurrent_state = trace.tensor(prefix + ":recurrent-state",
            (state_pool_slots, self.heads, self.dim, self.dim), "fp32", persistent=True)
        self.indices = trace.tensor(prefix + ":state-indices", (len(state_indices),), "int32")
        self.stages = {stage: SymbolicK3Stage(trace, prefix + ":" + stage, pool,
            variant="kimi_k3_kda_decode", shape={"rows": rows, "heads": self.heads,
                "head_dim": self.dim, "elements": rows * self.width}, attributes={
                    "source_component": "vllm_0_30_kimi_k3_triton_decode", "stage": stage,
                    "state_indices": state_indices, "state_pool_slots": state_pool_slots,
                    "gate_lower_bound": source.gate_lower_bound})
            for stage in ("convolution", "recurrent", "output_copy", "output_norm")}

    def _forward(self, *, mixed_qkv, g1, g2, beta, core_attn_out):
        # source non_spec_token_start=None: packed recurrence returns fresh
        # output, assignment copies into core, then gated norm aliases core.
        packed_conv_out = self.stages["convolution"](mixed_qkv, self.conv_state, self.indices,
            shape=mixed_qkv.shape, writes=(self.conv_state,))
        recurrent_output = self.stages["recurrent"](packed_conv_out, g1, beta, self.recurrent_state, self.indices,
            shape=core_attn_out.shape, writes=(self.recurrent_state,))
        self.stages["output_copy"](recurrent_output, shape=core_attn_out.shape, out=core_attn_out)
        self.stages["output_norm"](core_attn_out, g2, shape=core_attn_out.shape, out=core_attn_out,
            workspace=(("rstd", (self.rows * self.heads,), "fp32"),))

    def __call__(self, hidden_states, positions):
        num_tokens = hidden_states.shape[0]
        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        split_sizes = [3 * self.width, self.width, self.dim, self.heads]
        if self.padding:
            split_sizes.append(self.padding)
        projected = projected_qkvgfab.split(split_sizes, dim=-1)
        mixed_qkv, g_proj_states, f_a, beta = projected[:4]
        g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = g1.view_as_shape(1, num_tokens, self.heads, self.dim)
        g2 = g_proj_states.view_as_shape(num_tokens, self.heads, self.dim)
        core_attn_out = self.trace.tensor(self.prefix + ":core-out", (1, num_tokens, self.heads, self.dim), "bf16")
        self._forward(mixed_qkv=mixed_qkv, g1=g1, g2=g2, beta=beta, core_attn_out=core_attn_out)
        core_attn_out = core_attn_out.view_as_shape(num_tokens, self.width)
        return self.o_proj(core_attn_out)[0]


class KimiK3KDAPrefill(KimiK3KDADecode):
    """Fresh or mixed chunk program, using the source's three conv calls."""

    def __init__(self, trace, model, pool, prefix, *, query_lengths, initial_flags, **inputs):
        super().__init__(trace, model, pool, prefix, **inputs)
        self.sequences = len(query_lengths)
        starts = [0]
        for length in query_lengths:
            starts.append(starts[-1] + length)
        self.query_start = trace.tensor(prefix + ":query-start", (self.sequences + 1,), "int32")
        self.initial_mask = trace.tensor(prefix + ":has-initial-state", (self.sequences,), "bool")
        phase = "fresh_prefill" if self.sequences == 1 else "mixed"
        attributes = self.stages["convolution"].attributes
        self.stages = {stage: SymbolicK3Stage(trace, prefix + ":" + stage, pool,
            variant="kimi_k3_kda_prefill" if self.sequences == 1 else "kimi_k3_kda_mixed",
            shape={"rows": self.rows, "heads": self.heads,
                "head_dim": self.dim, "elements": self.sequences * self.heads * self.dim**2
                    if stage in ("initial_state", "final_state_scatter") else self.rows * self.width},
            attributes={**attributes, "source_component": "vllm_0_30_kimi_k3_triton_" + phase, "stage": stage,
                "query_start_loc": tuple(starts), "has_initial_state": initial_flags})
            for stage in ("q_convolution", "k_convolution", "v_convolution", "initial_state", "chunk",
                "final_state_scatter", "output_copy", "output_norm")}

    def _prefill_conv(self, x, state, stage):
        # empty_like compacts the non-dense [channel,token] input into its
        # column-major layout. The wrapper's transpose is dense token-major;
        # use one backing and expose the kernel's transposed output view.
        output = self.trace.tensor(self.trace.unique_name(self.prefix + ":" + stage + ":out"),
            (self.rows, self.width), "bf16")
        self.stages[stage](x.transpose(0, 1), state, self.indices, self.query_start, self.initial_mask,
            shape=(self.width, self.rows), out=output.transpose(0, 1), writes=(state,))
        return output

    def _forward(self, *, mixed_qkv, g1, g2, beta, core_attn_out):
        q_ns, k_ns, v_ns = mixed_qkv.split((self.width,) * 3)
        q_state = self.conv_state.narrow(1, 0, self.width)
        k_state = self.conv_state.narrow(1, self.width, self.width)
        v_state = self.conv_state.narrow(1, 2 * self.width, self.width)
        q_ns = self._prefill_conv(q_ns, q_state, "q_convolution")
        k_ns = self._prefill_conv(k_ns, k_state, "k_convolution")
        v_ns = self._prefill_conv(v_ns, v_state, "v_convolution")
        q_ns, k_ns, v_ns = (x.view_as_shape(1, self.rows, self.heads, self.dim) for x in (q_ns, k_ns, v_ns))
        initial = self.stages["initial_state"](self.recurrent_state, self.indices, self.initial_mask,
            shape=(self.sequences, self.heads, self.dim, self.dim), dtype="fp32")
        stage = self.stages["chunk"]
        node = stage.node(WorkloadKind.VECTOR, stage.shape, "bf16", stage.variant,
            {**stage.attributes, "symbolic_workspace_complete": False})
        final_state = self.trace.tensor(self.trace.unique_name(self.prefix + ":chunk-final-state"), initial.shape, "fp32")
        # chunk_gla_fwd_o_gk uses o=v when out is None. The wrapper's
        # contiguous V already has dense storage; its output is an alias.
        output = self.trace.call(node, (q_ns, k_ns, v_ns, g1, beta, initial, self.query_start),
            shape=v_ns.shape, dtype="bf16", out=v_ns, writes=(final_state,))
        self.stages["final_state_scatter"](final_state, self.indices, shape=self.recurrent_state.shape,
            dtype="fp32", out=self.recurrent_state)
        self.stages["output_copy"](output, shape=core_attn_out.shape, out=core_attn_out, controls=(self.recurrent_state,))
        self.stages["output_norm"](core_attn_out, g2, shape=core_attn_out.shape, out=core_attn_out,
            workspace=(("rstd", (self.rows * self.heads,), "fp32"),))


class KimiK3NoPEDecode:
    """NoPE MLA: fused norm, absorbed Q, fused cache/concat and V-up out=."""

    def __init__(self, trace, model, pool, prefix, *, rows, kv_lens, slots, cache_capacity_tokens):
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.heads, self.rows = model.num_attention_heads, rows
        self.split_frontend = rows < 512
        qkv = model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim
        gate = self.heads * model.v_head_dim
        self.packed = SymbolicLinear(trace, prefix + ":qkv-a", pool, model.hidden_size,
            qkv if self.split_frontend else qkv + gate)
        self.gate_proj = SymbolicLinear(trace, prefix + ":gate-proj", pool, model.hidden_size, gate)
        self.q_b_proj = SymbolicLinear(trace, prefix + ":q-b", pool, model.q_lora_rank, self.heads * model.head_dim)
        self.absorb = SymbolicBatchedLinear(trace, prefix + ":absorb", pool,
            self.heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.up = SymbolicBatchedLinear(trace, prefix + ":v-up", pool,
            self.heads, model.kv_lora_rank, model.v_head_dim)
        self.o_proj = SymbolicLinear(trace, prefix + ":out", pool, gate, model.hidden_size)
        self.cache = trace.tensor(prefix + ":cache", (cache_capacity_tokens, 576), "bf16", persistent=True)
        self.slots = trace.tensor(prefix + ":slots", (rows,), "int64")
        self.attention = SymbolicMLALatentDecode(trace, prefix + ":attention", pool,
            heads=self.heads, kv_lens=kv_lens, cache=self.cache)
        self.shape = {"rows": rows, "heads": self.heads, "q_lora_rank": model.q_lora_rank,
            "kv_lora_rank": model.kv_lora_rank, "elements": rows * (model.q_lora_rank + model.kv_lora_rank)}
        self.attrs = {"source_component": "vllm_0_30_kimi_k3_nope_decode", "slot_mapping": slots,
            "cache_capacity_tokens": cache_capacity_tokens, "rms_norm_eps": model.kimi_k3_semantics.rms_norm_eps}
        self.variant = "kimi_k3_nope_decode"

    def _stage(self, name, width, *, rows=None, slot_mapping=None):
        rows = self.rows if rows is None else rows
        attributes = {**self.attrs, "stage": name}
        if slot_mapping is not None:
            attributes["slot_mapping"] = slot_mapping
        return SymbolicK3Stage(self.trace, self.prefix + ":" + name, self.pool,
            variant=self.variant, shape={**self.shape, "rows": rows, "elements": rows * width}, attributes=attributes)

    def _apply_q_lora_attention(self, positions, hidden_states, qkv_lora):
        model, rows = self.model, hidden_states.shape[0]
        q_c, kv_c, k_pe = qkv_lora.split([model.q_lora_rank, model.kv_lora_rank, model.qk_rope_head_dim], dim=-1)
        norm = self._stage("q_kv_norm", model.q_lora_rank + model.kv_lora_rank)
        node = norm.node(WorkloadKind.VECTOR, norm.shape, "bf16", norm.variant,
            {**norm.attributes, "symbolic_workspace_complete": False})
        q_c, kv_c_normed = self.trace.call_outputs(node, (q_c, kv_c), outputs=(
            ("q", q_c.shape, "bf16"), ("kv", kv_c.shape, "bf16")))
        q = self.q_b_proj(q_c)[0].view_as_shape(rows, self.heads, model.head_dim)
        attn_out = self.trace.tensor(self.prefix + ":attn-out", (rows, self.heads * model.v_head_dim), "bf16")
        self._attention(positions, q, kv_c_normed, k_pe.unsqueeze(1), attn_out)
        return attn_out

    def _attention(self, positions, q, kv_c_normed, k_pe, attn_out, *, controls=()):
        model, rows = self.model, q.shape[0]
        q_nope, q_pe = q.split([model.qk_nope_head_dim, model.qk_rope_head_dim], dim=-1)
        ql_nope = self.absorb(q_nope.transpose(0, 1), controls=controls).transpose(0, 1)
        slot_tensor = self.slots if rows == self.rows else self.slots.narrow(0, 0, rows)
        mqa_q = self._stage("query_cache_concat", self.heads * 576, rows=rows,
            slot_mapping=self.attrs["slot_mapping"][:rows])(
            ql_nope, q_pe, kv_c_normed, k_pe, slot_tensor,
            shape=(rows, self.heads, 576), writes=(self.cache,))
        latent_out, lse = self.attention(mqa_q)
        self.up(latent_out.transpose(0, 1),
            out=attn_out.view_as_shape(rows, self.heads, model.v_head_dim).transpose(0, 1))

    def __call__(self, positions, hidden_states):
        model = self.model
        if self.split_frontend:
            qkv_lora = self.packed(hidden_states)[0]
            attn_out = self._apply_q_lora_attention(positions, hidden_states, qkv_lora)
            gate = self.gate_proj(hidden_states)[0]
        else:
            qkv_lora, gate = self.packed(hidden_states)[0].split([
                model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim,
                self.heads * model.v_head_dim], dim=-1)
            attn_out = self._apply_q_lora_attention(positions, hidden_states, qkv_lora)
        attn_out = self._stage("output_gate", self.heads * model.v_head_dim)(
            attn_out, gate, shape=attn_out.shape)
        return self.o_proj(attn_out)[0]


class KimiK3NoPEFreshPrefill(KimiK3NoPEDecode):
    """Fresh BF16 NoPE wrapper with the declared supports_out prefill branch."""

    def __init__(self, trace, model, pool, prefix, **inputs):
        super().__init__(trace, model, pool, prefix, **inputs)
        self.kv_b_proj = SymbolicLinear(trace, prefix + ":kv-b", pool, model.kv_lora_rank,
            model.num_attention_heads * (model.qk_nope_head_dim + model.v_head_dim))
        self.attrs = {**self.attrs, "source_component": "vllm_0_30_kimi_k3_nope_fresh_prefill"}
        self.variant = "kimi_k3_nope_prefill"

    def _attention(self, positions, q, kv_c_normed, k_pe, attn_out):
        self._prefill(q, kv_c_normed, k_pe, attn_out, offset=0)

    def _prefill(self, q, kv_c_normed, k_pe, attn_out, *, offset):
        model, rows = self.model, q.shape[0]
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view_as_shape(
            rows, self.heads, model.qk_nope_head_dim + model.v_head_dim)
        k_nope, v = kv_nope.split((model.qk_nope_head_dim, model.v_head_dim))
        slot_tensor = self.slots if rows == self.rows else self.slots.narrow(0, offset, rows)
        k = self._stage("key_cache_concat", self.heads * model.head_dim + 576, rows=rows,
            slot_mapping=self.attrs["slot_mapping"][offset:offset + rows])(
            q, k_nope, k_pe, kv_c_normed, slot_tensor, shape=q.shape, writes=(self.cache,))
        from .workload_graph import WorkloadNode
        node = WorkloadNode(self.trace.unique_name(self.prefix + ":prefill-attention"), WorkloadKind.FLASH_ATTN,
            self.pool, {"batch": 1, "q_len": rows, "kv_len": rows, "num_query_heads": self.heads,
                "qk_head_dim": model.head_dim, "v_head_dim": model.v_head_dim, "kv_lora_rank": model.kv_lora_rank},
            dtype="bf16", variant="mla_prefill", rank=self.trace.rank, attributes={
                "source_component": "vllm_0_30_symbolic_k3_prefill_attention", "launch_policy": "analytical_launch",
                "causal": True, "kv_dist": "uniform", "kv_layout": "mla", "q_lens": (rows,), "kv_lens": (rows,),
                "full_cuda_graphs": False, "max_cudagraph_capture_size": 0,
                "prefill_output_source": "supports_out", "return_softmax_lse": False,
                "mathematical_program": "bf16_mla_prefill_softmax_v1", "symbolic_workspace_complete": False})
        self.trace.call(node, (q, k, v), shape=(rows, self.heads, model.v_head_dim), dtype="bf16",
            out=attn_out.view_as_shape(rows, self.heads, model.v_head_dim), controls=(self.cache,))


class KimiK3NoPEMixed(KimiK3NoPEFreshPrefill):
    def __init__(self, trace, model, pool, prefix, **inputs):
        super().__init__(trace, model, pool, prefix, **inputs)
        self.attrs = {**self.attrs, "source_component": "vllm_0_30_kimi_k3_nope_mixed"}
        self.variant = "kimi_k3_nope_mixed"
        self.attention = SymbolicMLALatentDecode(trace, prefix + ":attention", pool,
            heads=self.heads, kv_lens=inputs["kv_lens"][:1], cache=self.cache)

    def _attention(self, positions, q, kv_c_normed, k_pe, attn_out):
        prefill_out = attn_out.narrow(0, 1, self.rows - 1)
        self._prefill(q.narrow(0, 1, self.rows - 1), kv_c_normed.narrow(0, 1, self.rows - 1),
            k_pe.narrow(0, 1, self.rows - 1), prefill_out, offset=1)
        KimiK3NoPEDecode._attention(self, positions.narrow(0, 0, 1), q.narrow(0, 0, 1),
            kv_c_normed.narrow(0, 0, 1), k_pe.narrow(0, 0, 1), attn_out.narrow(0, 0, 1), controls=(prefill_out,))


class KimiK3DecoderLayer:
    def __init__(self, trace, model, pool, prefix, *, layer, attention, resolution, invocation_id):
        size = model.kimi_k3_semantics.attention_residual_block_size
        self.is_block_write_layer = layer % size == 0
        blocks = -(-layer // size)
        self.pre = SymbolicK3AttentionResidual(trace, prefix + ":pre-attn", pool, model,
            blocks=blocks, write=layer // size if self.is_block_write_layer else -1)
        self.post = SymbolicK3AttentionResidual(trace, prefix + ":post-attn", pool, model,
            blocks=blocks + self.is_block_write_layer)
        self.self_attn = attention
        self.mlp = (KimiK3MoE(trace, model, pool, prefix + ":moe", resolution=resolution,
            invocation_id=invocation_id, layer_index=layer) if model.layer_uses_moe(layer)
            else KimiK3MLP(trace, model, pool, prefix + ":mlp"))

    def __call__(self, positions, hidden_states, residual, prefix_sum):
        hidden_states = self.pre(prefix_sum, hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        if self.is_block_write_layer:
            prefix_sum = hidden_states
            prefix_delta = None
        else:
            prefix_delta = hidden_states
        hidden_states = self.post(prefix_sum, prefix_delta, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, prefix_sum, residual


def trace_kimi_k3_forward(model, step, pool, expert_routing, *, input_ids, positions,
        slot_mapping, cache_capacity_tokens, state_indices, state_pool_slots, prefill_output_source=None):
    """Full93-layer TP1 decode or fresh graph with declared state/KV slots.

    Source branch: ordinary Triton KDA, BF16 MLA, modular MXFP4 Marlin,
    no PP/SP/CP/EP/MTP or auxiliary hidden capture. Stream/compiler/library
    services and unknown callee scratch remain outside this source graph.
    """
    config, source = pool.parallel_plan.config, model.kimi_k3_semantics
    if (model.model_type != "kimi_linear" or source is None or pool.hardware_id != "nvidia_h200"
            or model.compute_dtype != "bf16" or not source.latent_moe_use_norm or not source.mla_use_output_gate
            or (model.qk_nope_head_dim, model.qk_rope_head_dim, model.kv_lora_rank, model.v_head_dim) != (128, 64, 512, 128)
            or source.kda_head_dim != 128 or source.short_conv_kernel_size != 4
            or config.enable_expert_parallel or any(value != 1 for value in (config.tensor_parallel_size,
                config.pipeline_parallel_size, config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("virtual K3 requires its BF16 hybrid TP/PP/SP/CP/DP1 EP-off source")
    if (step.pool_name != pool.name or not step.requests or any(r.mtp_proposed_tokens or r.mtp_proposer_tokens or r.mtp_verified_tokens
            for r in step.requests)):
        raise NotImplementedError("virtual K3 requires ordinary nonempty target-off source requests")
    decode = all(r.phase == "decode" and r.scheduled_tokens == 1 and r.computed_tokens > 0 for r in step.requests)
    fresh = (len(step.requests) == 1 and step.requests[0].phase == "prefill" and step.requests[0].computed_tokens == 0
        and step.requests[0].scheduled_tokens >= 512 and step.requests[0].scheduled_tokens % 64 == 0)
    mixed = (len(step.requests) == 2 and step.requests[0].phase == "decode"
        and step.requests[0].scheduled_tokens == 1 and step.requests[0].computed_tokens > 0
        and step.requests[1].phase == "prefill" and step.requests[1].computed_tokens == 0
        and step.requests[1].scheduled_tokens >= 512 and step.requests[1].scheduled_tokens % 64 == 0)
    if not decode and not fresh and not mixed:
        raise NotImplementedError("virtual K3 supports single-token decode, fresh full64-chunk prefill or decode-first/fresh mixed")
    if ((fresh or mixed) and prefill_output_source != "supports_out") or (decode and prefill_output_source is not None):
        raise ValueError("K3 fresh prefill requires an explicit supports_out backend source declaration")
    rows = sum(r.scheduled_tokens for r in step.requests)
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(value) is not int for value in values):
            raise ValueError(f"K3 {name} requires one explicit integer per source row")
    if any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("K3 input ID is outside the accurate vocabulary")
    if positions != tuple(p for r in step.requests for p in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens)) or any(value >= model.max_position_embeddings for value in positions):
        raise ValueError("K3 positions must match the declared source contexts")
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slot_mapping)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slot_mapping)):
        raise ValueError("K3 logical KV pool requires disjoint valid write slots and all contexts")
    if (type(state_pool_slots) is not int or state_pool_slots <= 1
            or not isinstance(state_indices, tuple) or len(state_indices) != len(step.requests)
            or any(type(index) is not int or not 0 < index < state_pool_slots for index in state_indices)
            or len(set(state_indices)) != len(step.requests)):
        raise ValueError("K3 requires explicit distinct positive KDA state slots")
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
    trace = ForwardTrace(rank=pool.ranks[0], namespace=step.step_id)

    def forward():
        # Constructor objects own the same cache/program buffers throughout
        # forward. There is no manually entered edge or release schedule.
        layers = []
        for layer, kind in enumerate(source.layer_attention_types):
            prefix = f"layer-{layer}"
            base = f"language_model.model.layers.{layer}"
            projections = (("self_attn.in_proj_qkvgfab", "self_attn.f_b_proj", "self_attn.o_proj")
                if kind == "kda" else ("self_attn.fused_qkv_a_g_proj", "self_attn.q_b_proj", "self_attn.o_proj"))
            projections += (("block_sparse_moe.gate", "block_sparse_moe.routed_expert_down_proj",
                "block_sparse_moe.routed_expert_up_proj", "block_sparse_moe.shared_experts.gate_up_proj",
                "block_sparse_moe.shared_experts.down_proj") if model.layer_uses_moe(layer)
                else ("mlp.gate_up_proj", "mlp.down_proj"))
            if any(model.weight_dtype_for_module(base + "." + name, "Linear") != "bf16" for name in projections):
                raise NotImplementedError("K3 source projections require checkpoint BF16 module weights")
            if kind == "kda":
                attention_type = KimiK3KDADecode if decode else KimiK3KDAPrefill
                attention = attention_type(trace, model, pool.name, prefix + ":attention", rows=rows,
                    state_indices=state_indices, state_pool_slots=state_pool_slots,
                    **({} if decode else {"query_lengths": tuple(r.scheduled_tokens for r in step.requests),
                        "initial_flags": False if fresh else (True, False)}))
            elif kind == "mla_nope":
                if not decode and model.weight_dtype_for_module(base + ".self_attn.kv_b_proj", "Linear") != "bf16":
                    raise NotImplementedError("K3 prefill KV expansion requires BF16 checkpoint weights")
                attention_type = KimiK3NoPEDecode if decode else KimiK3NoPEFreshPrefill if fresh else KimiK3NoPEMixed
                attention = attention_type(trace, model, pool.name, prefix + ":attention", rows=rows,
                    kv_lens=kv_lens, slots=slot_mapping, cache_capacity_tokens=cache_capacity_tokens)
            else:
                raise NotImplementedError("K3 source has an unadapted attention type")
            layers.append(KimiK3DecoderLayer(trace, model, pool.name, prefix, layer=layer, attention=attention,
                resolution=resolutions[layer], invocation_id=invocations[layer]))
        if (model.weight_dtype_for_module("language_model.model.embed_tokens", "Embedding") != "bf16"
                or model.weight_dtype_for_module("language_model.lm_head", "Linear") != "bf16"):
            raise NotImplementedError("K3 source embedding and head require BF16 checkpoint storage")
        ids = trace.tensor("input_ids", (rows,), "int32")
        pos = trace.tensor("positions", (rows,), "int64")
        hidden_states = SymbolicEmbedding(trace, "embedding", pool.name, model.hidden_size, "bf16")(ids)
        capacity = -(-model.num_hidden_layers // source.attention_residual_block_size)
        residual = trace.tensor("block-residual", (rows, capacity, model.hidden_size), "bf16")
        prefix_sum, hidden_states = hidden_states, None
        for layer in layers:
            hidden_states, prefix_sum, residual = layer(pos, hidden_states, residual, prefix_sum)
        hidden_states = SymbolicK3AttentionResidual(trace, "output-attn-res", pool.name, model,
            blocks=capacity, output_norm=False)(prefix_sum, hidden_states, residual)
        selected_rows = tuple(sum(r.scheduled_tokens for r in step.requests[:i + 1]) - 1 for i in range(len(step.requests)))
        indices = trace.tensor("logits-indices", (len(selected_rows),), "int32")
        sample_hidden = SymbolicLogitsRowSelect(trace, "logits-row-select", pool.name)(
            hidden_states, indices, selected_indices=selected_rows)
        sample_hidden = SymbolicRMSNorm(trace, "final-norm", pool.name, model.hidden_size, source.rms_norm_eps)(sample_hidden)
        logits = SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
            model.vocab_size, kind=WorkloadKind.LM_HEAD)(sample_hidden)[0]
        sampled = SymbolicGreedy(trace, "target-sampler", pool.name)(logits)

    forward()
    return trace


class DeepseekV2KimiDecoderLayer(Qwen3DecoderLayer):
    """Reuse the BF16/no-SP residual decoder forward body."""

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index, attention, resolution, invocation_id,
                 expert_parallel_size=1):
        self.remote_ffn = None
        self.input_layernorm = SymbolicRMSNorm(trace, prefix + ":input-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        self.self_attn = attention
        self.post_attention_layernorm = SymbolicRMSNorm(trace, prefix + ":post-attn-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        if model.layer_uses_moe(layer_index):
            self.mlp = DeepseekV2KimiMoE(trace, model, pool, prefix + ":mlp", group, config,
                layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
                expert_parallel_size=expert_parallel_size)
        else:
            self.mlp = DeepseekV2MLP(trace, model, pool, prefix + ":mlp", intermediate_size=model.intermediate_size,
                tp_size=1 if group is None else len(group.traces), module_path=f"language_model.model.layers.{layer_index}.mlp",
                reduce_results=group is not None, reduction_group=group, reduction_config=config)


class DeepseekV2KimiModel(Qwen3Model):
    """K2.6 decoder or PP stage, sharing the same source forward."""

    def __init__(self, trace, model, pool, group, config, *, token_ids, q_lens, kv_lens, slots, cache_capacity_tokens,
                 resolutions, invocation_ids, model_input_order, num_decodes, expert_parallel_size=1,
                 context_chunk_rows=None, layer_range=None, physical_rows=None):
        self._source_module_program = None
        start, end = (0, model.num_hidden_layers) if layer_range is None else layer_range
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= model.num_hidden_layers):
            raise ValueError("Kimi stage requires a nonempty contiguous checkpoint layer range")
        self.first_stage, self.last_stage = start == 0, end == model.num_hidden_layers
        tp_size = 1 if group is None else len(group.traces)
        self.embed_tokens = None
        if self.first_stage:
            self.embed_tokens = (SymbolicEmbedding(trace, "embedding", pool, model.hidden_size, "bf16",
                model_input_order=model_input_order, vocab_size=model.vocab_size) if group is None else
                SymbolicVocabParallelEmbedding(trace, "embedding", pool, model.hidden_size,
                    model.vocab_size, group, config, token_ids=token_ids,
                    model_input_order=model_input_order if trace.rank == group.participants[0] else None))
        self.rope_cache = trace.tensor("model.rope-cache", (model.max_position_embeddings,
            model.qk_rope_head_dim), "bf16", persistent=True)
        rows = sum(q_lens)
        physical_rows = rows if physical_rows is None else physical_rows
        self.slot_mapping = trace.tensor("slot_mapping", (physical_rows,), "int64", persistent=True)
        slot_mapping = self.slot_mapping if physical_rows == rows else self.slot_mapping.narrow(0, 0, rows)
        self.layers = []
        for layer in range(start, end):
            prefix = f"layer-{layer}"
            cache = trace.tensor(prefix + ":kv-cache", (cache_capacity_tokens,
                model.kv_lora_rank + model.qk_rope_head_dim), "bf16", persistent=True)
            if num_decodes < len(q_lens):
                attention = DeepseekV2MLAMixedAttention(trace, model, pool, prefix + ":attention",
                    layer_index=layer, tp_size=tp_size, q_lens=q_lens, kv_lens=kv_lens,
                    num_decodes=num_decodes, cache=cache,
                    slot_mapping=slot_mapping, slots=slots, rotary_cache=self.rope_cache,
                    reduction_group=group, reduction_config=config, context_chunk_rows=context_chunk_rows)
            else:
                attention = DeepseekV2MLADecodeAttention(trace, model, pool, prefix + ":attention", layer_index=layer,
                    tp_size=tp_size, kv_lens=kv_lens, cache=cache, slot_mapping=slot_mapping, slots=slots,
                    rotary_cache=self.rope_cache, reduction_group=group, reduction_config=config,
                    backend_source="vllm_0_30_flashmla_bf16", q_lens=q_lens)
            self.layers.append(DeepseekV2KimiDecoderLayer(trace, model, pool, prefix, group, config,
                layer_index=layer, attention=attention, resolution=resolutions[layer], invocation_id=invocation_ids[layer],
                expert_parallel_size=expert_parallel_size))
        self.norm = (SymbolicRMSNorm(trace, "final-norm", pool, model.hidden_size, model.target_semantics.rms_norm_eps)
            if self.last_stage else None)


def _kimi_single_rank_source_scope(model, pool):
    config = pool.parallel_plan.config
    return (model.model_id == "moonshotai/Kimi-K2.6" and model.model_type == "kimi_k2"
        and model.compute_dtype == "bf16" and model.weight_dtype == "mixed_int4_bf16"
        and pool.hardware_id in ("nvidia_a800", "nvidia_h200") and len(pool.ranks) == 1
        and config.is_rank_local and not config.enable_expert_parallel
        and all(value == 1 for value in (config.tensor_parallel_size, config.pipeline_parallel_size,
            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size)))


def _kimi_pipeline_source_scope(model, pool):
    config = pool.parallel_plan.config
    return (model.model_id == "moonshotai/Kimi-K2.6" and model.model_type == "kimi_k2"
        and model.compute_dtype == "bf16" and model.weight_dtype == "mixed_int4_bf16"
        and pool.hardware_id == "nvidia_a800" and config.pipeline_parallel_size == 2
        and config.tensor_parallel_size == config.expert_parallel_size == 8 and config.enable_expert_parallel
        and config.is_rank_local and len(pool.ranks) == 16
        and all(value == 1 for value in (config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
        and pool.parallel_plan.layer_to_pipeline_stage == (0,) * 31 + (1,) * 30)


def trace_kimi_k2_forward(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
        cache_capacity_tokens, source_max_model_len, max_batched_tokens, source_profile, target_sampling_source=None,
        source_block_size=None, graph_token_shape=None, graph_eager_kinds=(), include_lm_head=False,
        source_prompt_lengths=None):
    """Full K2.6 ordinary target graph, optionally including greedy.

    Explicit modular/no-overlap, BF16 latent KV, MTP-off; actual provider,
    full workspace and normal capacity remain separate gates. Context-prefill
    covers one shared chunk, or single-request block-aligned continuations.
    """
    from .runtime_inputs import vllm_0_30_source_model_inputs
    config = pool.parallel_plan.config
    pipeline_source = _kimi_pipeline_source_scope(model, pool)
    single_rank = _kimi_single_rank_source_scope(model, pool)
    if type(include_lm_head) is not bool:
        raise TypeError("K2.6 include_lm_head must be boolean")
    if (source_profile != "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
            or model.model_id != "moonshotai/Kimi-K2.6" or model.model_type != "kimi_k2"
            or model.compute_dtype != "bf16" or model.weight_dtype != "mixed_int4_bf16"
            or not (single_rank or pipeline_source or pool.hardware_id == "nvidia_h200"
                and config.pipeline_parallel_size == 1 and config.tensor_parallel_size == 8)
            or any(value != 1 for value in (config.sequence_parallel_size,
                config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("virtual K2.6 requires exact checkpoint, NVIDIA TP1/PP1 EP-off, H200 TP8/PP1 EP-off/EP8 or A800 TP8/PP2/EP8, SP/CP/DP1")
    ordinary = bool(step.requests) and all(
        (r.phase == "prefill" and r.computed_tokens >= 0 and r.scheduled_tokens > 0)
        or (r.phase == "decode" and r.scheduled_tokens > 0 and r.computed_tokens > 0)
        for r in step.requests)
    if (step.pool_name != pool.name or not ordinary
            or any(r.mtp_proposed_tokens or r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests)):
        raise NotImplementedError("virtual K2.6 requires explicit positive decode/prefill queries with MTP execution off")
    if target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16"):
        raise NotImplementedError("virtual K2.6 target requires the declared BF16 plain greedy chain")
    rows = sum(r.scheduled_tokens for r in step.requests)
    physical_rows = rows
    if graph_token_shape is not None:
        from .graph_padding import GraphTokenShape
        if (not isinstance(graph_token_shape, GraphTokenShape) or not single_rank
                or graph_token_shape.logical_tokens != rows
                or graph_token_shape.runtime_mode not in ("FULL", "PIECEWISE")
                or (graph_token_shape.runtime_mode == "FULL" and any(
                    request.phase != "decode" or request.scheduled_tokens != 1 for request in step.requests))):
            raise NotImplementedError("Kimi source capture requires TP1/PP1 and FULL single-token decode or PIECEWISE")
        physical_rows = graph_token_shape.model_input_tokens
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(value) is not int for value in values):
            raise ValueError(f"K2.6 {name} requires one explicit integer per source row")
    if any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("K2.6 token IDs must belong to the accurate checkpoint vocabulary")
    if positions != tuple(position for r in step.requests
            for position in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens)):
        raise ValueError("K2.6 positions must match the declared source contexts")
    step, input_ids, model_input_order = vllm_0_30_source_model_inputs(step, embedding_token_ids=input_ids, prompt_lengths=source_prompt_lengths,
        decode_threshold=128, require_uniform=True)
    order = model_input_order["token_row_permutation"]
    positions = tuple(positions[row] for row in order)
    slots = tuple(slot_mapping[row] for row in order)
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    num_decodes = model_input_order["model_attention_phases"].count("decode")
    context_requests = tuple(request for request in step.requests[num_decodes:] if request.computed_tokens > 0)
    context_chunk_rows = None
    if context_requests:
        if sum(request.computed_tokens for request in context_requests) > min(8 * source_max_model_len, 65536):
            if len(context_requests) != 1:
                raise NotImplementedError("virtual K2.6 multi-chunk context currently requires one nonempty request")
            if type(source_block_size) is not int or source_block_size <= 0 or 65536 % source_block_size:
                raise NotImplementedError("virtual K2.6 multi-chunk context requires an explicit source block size dividing 65536")
            context_chunk_rows = 65536
        # Arena IDs stay attached to requests through source batch reordering.
        # Each request's context starts immediately before its first new slot.
        cursor = 0
        context_ranges = []
        for request in step.requests:
            request_slots = slots[cursor:cursor + request.scheduled_tokens]
            start = request_slots[0] - request.computed_tokens
            if start < 0 or request_slots != tuple(range(start + request.computed_tokens,
                    start + request.computed_tokens + request.scheduled_tokens)):
                raise ValueError("K2.6 context requires the declared disjoint logical request arena, not native page IDs")
            context_ranges.append((start, start + request.computed_tokens + request.scheduled_tokens))
            cursor += request.scheduled_tokens
        ordered_ranges = sorted(context_ranges)
        if any(left[1] > right[0] for left, right in zip(ordered_ranges, ordered_ranges[1:])):
            raise ValueError("K2.6 context requires disjoint logical request arenas")
    if (type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
            or type(max_batched_tokens) is not int or max_batched_tokens < physical_rows):
        raise ValueError("K2.6 source context and batch limits must contain every row")
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slots)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slots)):
        raise ValueError("K2.6 logical cache must contain contexts and distinct valid update slots")
    stages = tuple(tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}")
        for rank in pool.physical_ranks_for_pipeline_stage(stage)) for stage in range(config.pipeline_parallel_size))
    groups = tuple(None if single_rank else ForwardCollectiveGroup(traces,
        namespace=f"{step.step_id}:stage-{stage}" if pipeline_source else step.step_id)
        for stage, traces in enumerate(stages))
    collective = None if single_rank else pool.collective_for("all_reduce")
    target_head = include_lm_head or target_sampling_source is not None
    gather = None if single_rank or not target_head else pool.collective_for("all_gather")
    invocation_ids = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocation_ids, physical_rows)

    def rank_forward(trace, stage, incoming):
        group = groups[stage]
        own_layers = tuple(index for index, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage) if owner == stage)
        network = DeepseekV2KimiModel(trace, model, pool.name, group, collective, token_ids=input_ids,
            q_lens=q_lens, kv_lens=kv_lens, slots=slots, cache_capacity_tokens=cache_capacity_tokens,
            resolutions=resolutions, invocation_ids=invocation_ids, model_input_order=model_input_order,
            num_decodes=num_decodes, expert_parallel_size=config.expert_parallel_size,
            context_chunk_rows=context_chunk_rows, layer_range=(own_layers[0], own_layers[-1] + 1),
            physical_rows=physical_rows)
        ids = trace.tensor("input_ids", (physical_rows,), "int32") if network.first_stage else None
        pos = trace.tensor("positions", (physical_rows,), "int64")
        indices = (trace.tensor("logits_indices", (len(q_lens),), "int32")
            if network.last_stage and target_head else None)
        if graph_token_shape is not None:
            for layer in network.layers:
                layer.self_attn.graph_token_shape = graph_token_shape
            trace.begin_capture(key=f"{source_profile}:{graph_token_shape.runtime_mode}:{physical_rows}",
                logical_tokens=rows, model_input_tokens=physical_rows,
                inputs={"input_ids": ids, "positions": pos, "slot_mapping": network.slot_mapping},
                runtime_mode=graph_token_shape.runtime_mode, eager_kinds=graph_eager_kinds)
        hidden_states = network(ids, pos, intermediate_tensors=incoming)
        if graph_token_shape is not None:
            trace.end_capture(hidden_states)
        if not network.last_stage:
            mark_source_scope(trace)
            return hidden_states
        if target_head:
            sampled = _plain_greedy_target(trace, model, pool.name, group, gather, hidden_states, indices, q_lens,
                sample_logits=target_sampling_source == "vllm_0_30_plain_greedy_bf16",
                cast_logits=target_sampling_source == "vllm_0_30_plain_sampler_bf16")
        mark_source_scope(trace)
        return None

    def mark_source_scope(trace):
        trace._nodes = {name: replace(node, attributes={**node.attributes,
            "source_projection_only": True, "model_execution_version": "vllm_0_30_0",
            "model_source_profile": source_profile}) for name, node in trace._nodes.items()}

    if pipeline_source:
        pipeline = ForwardPipelineGroup(tuple(trace for traces in stages for trace in traces),
            namespace=step.step_id, collective_groups=groups)
        pipeline.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
            "pipeline_layer_partition": (31, 30), "tp_partial_hidden_sent_without_added_reduction": False,
            "numerical_tp_pp_equivalence_bound": False, "normal_capacity_accepted": False, "actual_provider_bound": False}
        _run_vllm030_pipeline_stages(pipeline, pool, rows=rows, hidden_size=model.hidden_size,
            max_batched_tokens=max_batched_tokens, run_stage=rank_forward)
        return pipeline
    for trace in stages[0]:
        rank_forward(trace, 0, None)
    return stages[0][0] if single_rank else groups[0]


def _validate_qwen3_source(model):
    if (model.model_type != "qwen3" or model.compute_dtype not in ("bf16", "fp16")
            or model.weight_dtype != model.compute_dtype or model.num_experts
            or model.hidden_act != "silu"
            or model.rope_scaling_type != "default" or model.rms_norm_eps is None):
        raise NotImplementedError("symbolic complete model requires unquantized 16-bit default-RoPE Qwen3")


def trace_qwen3_forward(model: ModelConfig, step, *, input_ids, positions,
                        slot_mapping, cache_capacity_tokens, source_profile, rank=0, namespace=None,
                        hardware_id="nvidia_h200", model_execution_version="vllm_0_30_0", sample_logits=True, cast_logits=False,
                        maca_copy_runtime_source=None, ascend_piecewise_source=False):
    _validate_qwen3_source(model)
    if hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b"):
        raise NotImplementedError("dense CUDA-alike forward requires its supported source platform")
    return _trace_dense_forward(model, step, input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, rank=rank, namespace=namespace, model_type=Qwen3Model,
        rope_program=vllm_rope_program(hardware_id, model.rope_scaling_type), hardware_id=hardware_id,
        model_execution_version=model_execution_version, sample_logits=sample_logits, cast_logits=cast_logits,
        maca_copy_runtime_source=maca_copy_runtime_source, ascend_piecewise_source=ascend_piecewise_source)


def trace_llama_forward(model: ModelConfig, step, *, input_ids, positions,
                        slot_mapping, cache_capacity_tokens, source_profile, rank=0, namespace=None,
                        hardware_id="nvidia_h200", model_execution_version="vllm_0_30_0", sample_logits=True, cast_logits=False,
                        maca_copy_runtime_source=None):
    _validate_llama_source(model)
    if hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b"):
        raise NotImplementedError("dense CUDA-alike forward requires its supported source platform")
    return _trace_dense_forward(model, step, input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, rank=rank, namespace=namespace, model_type=LlamaModel,
        rope_program=vllm_rope_program(hardware_id, model.rope_scaling_type), hardware_id=hardware_id,
        model_execution_version=model_execution_version, sample_logits=sample_logits, cast_logits=cast_logits,
        maca_copy_runtime_source=maca_copy_runtime_source)


def _validate_llama_source(model):
    if (model.model_type != "llama" or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16" or model.num_experts
            or model.hidden_act != "silu" or model.attention_bias
            or model.rope_scaling_type not in ("default", "llama3") or model.rms_norm_eps is None):
        raise NotImplementedError("symbolic Llama requires dense BF16 SiLU with default/llama3 cached RoPE")


def trace_target_model_structural_forward(model, step, pool, expert_routing, *,
        hidden_shape, hidden_dtype, representative_layers):
    """A800 TP1 representative GLM/Kimi layers from a typed hidden port.

    Reuses the vLLM0.30 source modules without embedding, head or sampler.
    Selected layers form a representative source composition, not the omitted
    full decoder. GLM sparse decode shares the first layer's top-k buffer;
    fresh dense-MHA prefill does not consume it in the attention backend.
    Fresh short GLM prefill follows dense MHA, rather than forcing sparse MQA
    to match an older artifact. This is no0.27.1/runtime equivalence claim.
    """
    config = pool.parallel_plan.config
    targets = {"zai-org/GLM-5.2": (0, 3), "moonshotai/Kimi-K2.6": (0, 1)}
    if (model.model_id not in targets or representative_layers != targets[model.model_id]
            or model.compute_dtype != "bf16" or pool.hardware_id != "nvidia_a800"
            or config.tensor_parallel_size != 1 or len(pool.ranks) != 1
            or not config.is_rank_local or config.enable_expert_parallel):
        raise NotImplementedError("target structural source requires exact GLM5.2/K2.6 A800 TP1 representative layers")
    if (model.model_type not in ("glm_moe_dsa", "kimi_k2") or step.pool_name != pool.name
            or len(step.requests) != 1):
        raise NotImplementedError("target structural source requires one ordinary representative request")
    request = step.requests[0]
    if (not ((request.phase == "prefill" and request.computed_tokens == 0 and request.scheduled_tokens > 0)
            or (request.phase == "decode" and request.scheduled_tokens == 1 and request.computed_tokens > 0))
            or request.mtp_proposer_mode != "disabled" or request.mtp_proposed_tokens
            or request.mtp_proposer_tokens or request.mtp_verified_tokens or request.mtp_next_proposed_tokens):
        raise NotImplementedError("target structural source requires fresh prefill or single-token decode with MTP off")
    rows, capacity = request.scheduled_tokens, request.computed_tokens + request.scheduled_tokens
    if (hidden_shape != (rows, model.hidden_size) or hidden_dtype != "bf16"
            or not isinstance(hidden_shape, tuple) or any(type(size) is not int for size in hidden_shape)
            or capacity > model.max_position_embeddings):
        raise ValueError("target structural hidden port must match source rows, checkpoint width and BF16 dtype")
    if model.model_type == "glm_moe_dsa" and (model.weight_dtype != "bf16"
            or rows > model.index_topk or tuple(model.indexer_types[layer] for layer in representative_layers) != ("full", "shared")):
        raise NotImplementedError("GLM representative source requires BF16 full/shared indexer and short fresh prefill")
    trace = ForwardTrace(rank=pool.ranks[0], namespace=step.step_id)
    trace.source_scope = {"execution_scope": "representative_decoder_layers_only",
        "input_source": "captured_hidden_shape_dtype", "source_projection_only": True,
        "model_execution_version": "vllm_0_30_0", "representative_layers": representative_layers,
        "embedding_executed": False, "sampler_executed": False,
        "actual_provider_bound": False, "full_decoder_executed": False,
        "source_version_equivalence_claimed": False,
        "prefill_attention_source": "dense_mha_fresh" if request.phase == "prefill" else None}
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)

    def forward():
        slots = tuple(range(request.computed_tokens, capacity))
        slot_mapping = trace.tensor("slot_mapping", (rows,), "int64", persistent=True)
        rope = trace.tensor("rotary_cache", (model.max_position_embeddings, model.qk_rope_head_dim), "bf16", persistent=True)
        layers = []
        if model.model_type == "glm_moe_dsa":
            topk = trace.tensor("topk", (rows, model.index_topk), "int32", persistent=True)
            dummy_q = trace.tensor("dummy:index-q", (1, 1, 1), "bf16", persistent=True)
            dummy_weights = trace.tensor("dummy:index-weights", (1, 1), "fp32", persistent=True)
            workspace = ForwardWorkspace(trace, "worker.workspace", alignment_bytes=256,
                initial_capacity_bytes=0, locked=False)
        for layer in representative_layers:
            prefix = f"layer-{layer}"
            cache = trace.tensor(prefix + ":cache", (capacity, model.kv_lora_rank + model.qk_rope_head_dim),
                "bf16", persistent=True)
            if model.model_type == "glm_moe_dsa":
                index_cache = (trace.tensor(prefix + ":index-cache", (capacity, 132), "uint8", persistent=True)
                    if model.indexer_types[layer] == "full" else None)
                attention = DeepseekV32Attention(trace, model, pool.name, prefix + ":attention",
                    layer_index=layer, tp_size=1, q_lens=(rows,), kv_lens=(capacity,), request_phases=(request.phase,),
                    cache=cache, index_cache=index_cache, topk_buffer=topk, rotary_cache=rope, index_rotary_cache=rope,
                    slot_mapping=slot_mapping, dummy_index_q=dummy_q, dummy_index_weights=dummy_weights,
                    workspace=workspace, max_batched_tokens=rows, selection_source="declared_causal_topk",
                    source_max_model_len=capacity)
                decoder = DeepseekV32DecoderLayer(trace, model, pool.name, prefix, None, None,
                    layer_index=layer, self_attn=attention, resolution=resolutions[layer], invocation_id=invocations[layer])
            else:
                arguments = dict(layer_index=layer, tp_size=1, cache=cache, slot_mapping=slot_mapping,
                    slots=slots, rotary_cache=rope, reduction_group=None, reduction_config=None)
                attention = (DeepseekV2MLAFreshPrefillAttention(trace, model, pool.name, prefix + ":attention",
                    q_lens=(rows,), **arguments) if request.phase == "prefill" else
                    DeepseekV2MLADecodeAttention(trace, model, pool.name, prefix + ":attention",
                        kv_lens=(capacity,), backend_source="vllm_0_30_flashmla_bf16", **arguments))
                decoder = DeepseekV2KimiDecoderLayer(trace, model, pool.name, prefix, None, None,
                    layer_index=layer, attention=attention, resolution=resolutions[layer], invocation_id=invocations[layer])
            layers.append(decoder)
        hidden = trace.tensor("captured-hidden-input", hidden_shape, hidden_dtype)
        positions = trace.tensor("positions", (rows,), "int64")
        residual = None
        for layer in layers:
            hidden, residual = layer(positions, hidden, residual)

    forward()
    return trace


def trace_dense_structural_forward(model, step, pool, *, hidden_shape, hidden_dtype,
                                   model_execution_version, graph_token_shape=None, input_all_reduce=False,
                                   expert_routing=None):
    """Existing GQA decoder plus head from a captured typed hidden port.

    Qwen2Model.forward(inputs_embeds) and LlamaModel.forward(inputs_embeds)
    share this ordinary no-auxiliary body in local vLLM0.11 and0.30.
    Preserve the caller's source identity; this projection does not execute
    embedding, sampler, tensor values, a runtime or a native provider.
    Logical cache slots follow the observed ScheduleStep's disjoint requests.
    """
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_30_0"):
        raise NotImplementedError("structural dense forward requires its declared common source version")
    from .graph_padding import GraphTokenShape
    if graph_token_shape is not None and not isinstance(graph_token_shape, GraphTokenShape):
        raise NotImplementedError("structural dense forward requires explicit typed graph padding")
    parallel = pool.parallel_plan.config
    if model.model_type == "qwen3":
        _validate_qwen3_source(model)
        model_type = Qwen3Model
    elif model.model_type == "llama":
        _validate_llama_source(model)
        model_type = LlamaModel
    elif model.model_type == "qwen3_moe":
        _validate_qwen3_moe_source(model, parallel)
        model_type = Qwen3Model
    else:
        raise NotImplementedError("structural hidden-input forward requires its existing BF16 Qwen3/MoE or Llama source")
    if model.model_type != "qwen3_moe" and expert_routing is not None:
        raise ValueError("structural dense forward cannot consume expert routing")
    tp = parallel.tensor_parallel_size
    if (pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
            or tp not in (1, 2) or len(pool.ranks) != tp
            or parallel.pipeline_parallel_size != 1 or not parallel.is_rank_local
            or (parallel.enable_expert_parallel and model.model_type != "qwen3_moe")):
        raise NotImplementedError("structural hidden-input forward requires its existing TP1/TP2 PP/SP/CP/DP1 and Qwen3MoE EP2 scope")
    if type(input_all_reduce) is not bool or input_all_reduce != (tp == 2):
        raise ValueError("structural TP2 requires an explicit captured input all-reduce; TP1 does not")
    if (step.pool_name != pool.name or not step.requests
            or any(request.phase not in ("prefill", "decode") or request.scheduled_tokens <= 0
                or (request.phase == "decode" and request.scheduled_tokens != 1)
                or request.mtp_proposer_mode != "disabled" or request.mtp_proposed_tokens
                or request.mtp_proposer_tokens or request.mtp_verified_tokens or request.mtp_next_proposed_tokens
                for request in step.requests)):
        raise NotImplementedError("structural dense forward requires ordinary prefill/decode/mixed with MTP off")
    q_lens = tuple(request.scheduled_tokens for request in step.requests)
    kv_lens = tuple(request.computed_tokens + request.scheduled_tokens for request in step.requests)
    rows = sum(q_lens)
    physical_rows = rows
    if graph_token_shape is not None:
        if (model.model_type not in ("qwen3", "llama")
                or graph_token_shape.logical_tokens != rows
                or graph_token_shape.runtime_mode not in ("FULL", "PIECEWISE")):
            raise NotImplementedError("structural graph padding requires dense TP1/TP2 and matching logical rows")
        physical_rows = graph_token_shape.model_input_tokens
    if (not isinstance(hidden_shape, tuple) or any(type(dimension) is not int for dimension in hidden_shape)
            or hidden_shape != (physical_rows, model.hidden_size) or hidden_dtype != model.compute_dtype):
        raise ValueError("structural hidden input must match captured rows, model width and BF16 dtype")
    if max(kv_lens) > model.max_position_embeddings:
        raise ValueError("structural source contexts exceed the checkpoint rotary extent")
    slots, offset = [], 0
    for request in step.requests:
        slots.extend(range(offset + request.computed_tokens,
            offset + request.computed_tokens + request.scheduled_tokens))
        offset += request.computed_tokens + request.scheduled_tokens
    scope = {"execution_scope": "decoder_and_logits_only", "input_source": "captured_hidden_shape_dtype",
        "model_execution_version": model_execution_version, "source_projection_only": True,
        "embedding_executed": False, "sampler_executed": False, "actual_provider_bound": False,
        "captured_input_all_reduce": input_all_reduce, "logical_tokens": rows,
        "model_input_tokens": physical_rows, "graph_capture_executed": False}
    factories = None
    if model.model_type == "qwen3_moe":
        from functools import partial
        invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
        resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
        factories = tuple(partial(Qwen3MoeSparseMoeBlock, resolution=resolution, invocation_id=invocation,
            ep_size=parallel.expert_parallel_size) for resolution, invocation in zip(resolutions, invocations, strict=True))
        scope["moe_routing_math_source"] = "vllm_0_30_bf16_softmax_topk"
        scope["moe_router_native_source_equivalence_claimed"] = False
    traces = tuple(ForwardTrace(rank=rank, namespace=step.step_id if tp == 1 else f"{step.step_id}:rank-{rank}")
        for rank in pool.ranks)
    group = None if tp == 1 else ForwardCollectiveGroup(traces, namespace=step.step_id)
    selected_indices = tuple(sum(q_lens[:index + 1]) - 1 for index in range(len(q_lens)))

    def rank_forward(trace):
        trace.source_scope = dict(scope)
        network = model_type(trace, model, pool.name, q_lens, kv_lens, offset, tuple(slots),
            input_mode="inputs_embeds", group=group, mlp_factories=factories,
            collective=None if group is None else pool.collective_for("all_reduce"),
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type), hardware_id=pool.hardware_id)
        if graph_token_shape is not None:
            network.inputs_embeds_shape = hidden_shape
            slot_port = trace.tensor("captured-slot-mapping", (physical_rows,), "int64")
            for layer in network.layers:
                layer.self_attn.attn.graph_token_shape = graph_token_shape
                layer.self_attn.attn.slot_tensor = slot_port
        hidden = trace.tensor("captured-hidden-input", hidden_shape, hidden_dtype)
        if group is not None:
            hidden = group.call("embedding:all-reduce", hidden, pool_name=pool.name,
                config=pool.collective_for("all_reduce"), inplace=False)
        positions = trace.tensor("positions", (physical_rows,), "int64")
        hidden = network(None, positions, inputs_embeds=hidden)
        indices = trace.tensor("logits_indices", (len(q_lens),), "int32")
        selected = SymbolicLogitsRowSelect(trace, "logits-boundary-gather", pool.name)(
            hidden, indices, selected_indices=selected_indices, model_execution_version=model_execution_version)
        if group is None:
            logits, _ = SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
                model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected)
        else:
            logits = SymbolicTPLogitsProcessor(trace, "lm-head", pool.name, hidden_size=model.hidden_size,
                vocab_size=model.vocab_size, group=group, config=pool.collective_for("all_gather"),
                model_execution_version=model_execution_version)(selected)

    for trace in traces:
        rank_forward(trace)
    if group is not None:
        group.source_scope = dict(scope)
        return group
    return traces[0]


def trace_af_forward(model, step, deployment, expert_routing, *, input_ids, positions, slot_mapping,
                     cache_capacity_tokens, source_profile,
                     max_model_len=None, max_batched_tokens=None, max_requests=None,
                     model_execution_version="vllm_0_30_0", graph_token_shape=None, graph_eager_kinds=()):
    """Project one configured AF pair while executing the common model forward.

    Vanilla upstream vLLM supplies the local modules, not an AF runtime. The
    two full-tensor exchanges are the repository's existing AF service contract.
    """
    from functools import partial
    from src.serving.deployment.deployment_config import worker_role_operator_group, worker_role_phase
    if deployment.mode not in ("af_disagg", "pd_af_disagg"):
        raise NotImplementedError("virtual AF requires the existing standalone attention and FFN pools")
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_30_0"):
        raise ValueError("virtual AF requires its declared supported model version")
    if model_execution_version == "vllm_0_11_0" and model.model_type not in ("qwen3", "llama", "qwen3_moe", "glm4_moe"):
        raise NotImplementedError("virtual AF0.11 requires its existing Qwen/Llama or GLM4 source")
    if graph_token_shape is not None and (model_execution_version != "vllm_0_11_0"
            or model.model_type not in ("qwen3", "llama") or model.num_experts
            or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"):
        raise NotImplementedError("segmented AF capture requires its A8000.11 BF16 dense Qwen/Llama source")
    attention_pool = deployment.worker_pools[step.pool_name]
    if worker_role_operator_group(attention_pool.role) != "attention":
        raise ValueError("virtual AF steps must belong to the attention Scheduler")
    phase = worker_role_phase(attention_pool.role)
    routes = tuple(route for route in deployment.routes if route.payload == "activation"
        and route.source_pool == attention_pool.name)
    if len(routes) != 1:
        raise ValueError("virtual AF requires exactly one outgoing activation route")
    ffn_pool = deployment.worker_pools[routes[0].destination_pool]
    if (worker_role_operator_group(ffn_pool.role) != "ffn" or worker_role_phase(ffn_pool.role) != phase
            or sum(route.payload == "activation" and route.source_pool == ffn_pool.name
                and route.destination_pool == attention_pool.name for route in deployment.routes) != 1):
        raise ValueError("virtual AF requires a reciprocal same-phase FFN partner")
    pools = (attention_pool, ffn_pool)
    if deployment.mode == "af_disagg" and (len(deployment.worker_pools) != 2 or phase != "colocated"):
        raise NotImplementedError("virtual standalone AF requires its two-pool deployment")
    if deployment.mode == "pd_af_disagg":
        if (phase not in ("prefill", "decode")
                or any(pool.parallel_plan.config.tensor_parallel_size != 1
                    or pool.parallel_plan.config.enable_expert_parallel for pool in pools)):
            raise NotImplementedError("virtual PD+AF preserves its existing TP1/EP-off source scope")
        if any(request.phase != phase and not (phase == "decode" and request.phase == "prefill"
                and (request.resumed_from_preemption or request.recomputed_tokens > 0)) for request in step.requests):
            raise ValueError("PD+AF request phase must match its attention owner or explicit decode recompute")
    tp = attention_pool.parallel_plan.config.tensor_parallel_size
    if tp not in (1, 2) or any(pool.hardware_id != "nvidia_a800"
            or pool.parallel_plan.config.tensor_parallel_size != tp
            or pool.parallel_plan.config.pipeline_parallel_size != 1
            or not pool.parallel_plan.config.is_rank_local or len(pool.ranks) != tp for pool in pools):
        raise NotImplementedError("virtual AF requires A800, matching TP1/2 and PP/SP/CP/DP1")
    if attention_pool.parallel_plan.config.enable_expert_parallel:
        raise NotImplementedError("AF expert parallelism belongs to the FFN pool")
    ep = ffn_pool.parallel_plan.config.expert_parallel_size
    if ep > 1:
        from src.serving.deployment.validated_scope import is_validated_standalone_af_ep2_scope
        if not is_validated_standalone_af_ep2_scope(deployment, model):
            raise NotImplementedError("virtual AF EP2 requires the existing source-backed two-layer scope")
    factories = None
    if model.model_type == "qwen3":
        _validate_qwen3_source(model)
        model_type = Qwen3Model
    elif model.model_type == "llama":
        _validate_llama_source(model)
        model_type = LlamaModel
    elif model.model_type == "qwen3_moe":
        config = ffn_pool.parallel_plan.config
        _validate_qwen3_moe_source(model, config)
        if ep == 1 and tp != 1:
            raise NotImplementedError("ordinary virtual AF MoE preserves the existing TP1 domain")
        rows = sum(request.scheduled_tokens for request in step.requests)
        invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
        resolutions = _source_expert_resolutions(model, step, ffn_pool, expert_routing, invocations, rows)
        factories = tuple(partial(Qwen3MoeSparseMoeBlock, resolution=resolution, invocation_id=invocation,
            ep_size=ep, model_execution_version=model_execution_version)
            for resolution, invocation in zip(resolutions, invocations, strict=True))
        model_type = Qwen3Model
    elif model.model_type == "glm4_moe":
        _validate_glm4_source(model)
        if tp != 1 or ep != 1:
            raise NotImplementedError("virtual GLM4 AF preserves its A800 TP1/EP-off source domain")
        model_type = Glm4MoeModel
    else:
        raise NotImplementedError("virtual AF requires an adapted Qwen3, Qwen3MoE or Llama source model")
    traces = {pool.name: tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}")
        for rank in pool.ranks) for pool in pools}
    groups = {pool.name: ForwardCollectiveGroup(traces[pool.name], namespace=f"{step.step_id}:{pool.name}")
        for pool in pools} if tp > 1 else {}
    transport = ForwardActivationGroup(deployment, traces,
        collective_groups=tuple(groups.values()), graph_token_shape=graph_token_shape,
        graph_eager_kinds=graph_eager_kinds)
    factory = partial(RemoteFFN, attention_pool=attention_pool, ffn_pool=ffn_pool,
        transport=transport, group=None if tp == 1 else groups[ffn_pool.name])
    if model.model_type == "glm4_moe":
        source_inputs = dict(input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_profile=source_profile,
            trace=traces[attention_pool.name][0], ffn_factory=factory, expert_pool=ffn_pool)
        if any(r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests):
            trace_glm4_mtp_forward(model, step, attention_pool, expert_routing,
                max_model_len=max_model_len, max_batched_tokens=max_batched_tokens, max_requests=max_requests,
                **source_inputs)
        else:
            trace_glm4_forward(model, step, attention_pool, expert_routing, **source_inputs)
        return transport
    for trace in traces[attention_pool.name]:
        _trace_dense_forward(model, step, input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_profile=source_profile,
            rank=trace.rank, namespace=trace.namespace, model_type=model_type, trace=trace,
            group=None if tp == 1 else groups[attention_pool.name],
            collective=None if tp == 1 else attention_pool.collective_for("all_reduce"),
            gather=None if tp == 1 else attention_pool.collective_for("all_gather"),
            hardware_id=attention_pool.hardware_id, ffn_factory=factory, mlp_factories=factories,
            model_execution_version=model_execution_version,
            graph_token_shape=graph_token_shape, graph_eager_kinds=graph_eager_kinds,
            af_transport=transport,
            fused_embedding=model_execution_version != "vllm_0_11_0")
    if graph_token_shape is not None:
        for group in groups.values():
            _remap_collective_capture_nodes(group)
    return transport


def _run_vllm030_pipeline_stages(pipeline, pool, *, rows, hidden_size, max_batched_tokens, run_stage):
    """Shared sync tensor-dict receive/gather/copy and original TP lane send.

    Model callbacks own decoder execution and final logits; this routine only
    expresses the pinned0.30 runner/transport's lightweight tensor lifetimes.
    """
    from dataclasses import replace
    groups = pipeline.collective_groups
    stages = tuple(group.traces for group in pipeline.collective_groups)
    tp = len(stages[0])
    gather = pool.collective_for("all_gather")
    incoming = [None] * tp
    length = rows * hidden_size // tp
    for stage, traces in enumerate(stages):
        next_incoming = []
        for lane, trace in enumerate(traces):
            prepared = None
            if stage > 0:
                # recv_tensor_dict retains both full backings through its
                # postprocess gathers, then the runner copies into its buffers.
                wire = incoming[lane]
                gathered = {name: groups[stage].call("receive:" + name, wire[name], pool_name=pool.name,
                    config=gather, inplace=False, dim=0, output_shape=(rows, hidden_size))
                    for name in ("hidden_states", "residual")}
                incoming[lane] = None
                del wire
                prepared = {}
                for name in ("hidden_states", "residual"):
                    buffer = trace.tensor("worker.intermediate:" + name,
                        (max_batched_tokens, hidden_size), "bf16", persistent=True)
                    active = buffer if max_batched_tokens == rows else buffer.narrow(0, 0, rows)
                    prepared[name] = SymbolicCopy(trace, "receive:runner-copy:" + name, pool.name)(gathered[name], active)
                del gathered
            outputs = run_stage(trace, stage, prepared)
            del prepared
            if stage == len(stages) - 1:
                continue
            destination = stages[stage + 1][lane]
            wire = {}
            for name in ("hidden_states", "residual"):
                tensor = outputs[name]
                full = destination.tensor("receive:" + name, tensor.shape, tensor.dtype)
                received = full.view_as_shape(-1).narrow(0, lane * length, length)
                sent = tensor.view_as_shape(-1).narrow(0, lane * length, length)
                wire[name] = pipeline.send(sent, config=pool.pipeline_p2p_for(stage, stage + 1), pool_name=pool.name,
                    tensor_name=name, frontier=outputs["hidden_states"], out=received)
            next_incoming.append(wire)
            del outputs, tensor, full, received, sent, wire
        incoming = next_incoming
    for trace in pipeline.traces:
        for identifier, node in tuple(trace._nodes.items()):
            if "model_execution_version" in node.attributes and node.attributes["model_execution_version"] != "vllm_0_30_0":
                raise ValueError("pipeline node source version conflicts with its vLLM0.30 projection")
            trace._nodes[identifier] = replace(node, attributes={**node.attributes,
                "source_projection_only": True, "model_execution_version": "vllm_0_30_0"})


def trace_dense_pipeline_forward(model, step, pool, *, input_ids, positions, slot_mapping,
                                 cache_capacity_tokens, source_profile, model_execution_version="vllm_0_30_0",
                                 graph_token_shape=None, graph_eager_kinds=()):
    """Run the same Qwen3 module forward over contiguous source PP stages."""
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
    config = pool.parallel_plan.config
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_30_0"):
        raise ValueError("virtual PP requires its supported declared model version")
    _validate_qwen3_source(model)
    if (not is_validated_qwen3_pipeline_model_scope(model, config.tensor_parallel_size, config.pipeline_parallel_size)
            or pool.hardware_id != "nvidia_a800"
            or config.enable_expert_parallel or not config.is_rank_local
            or len(pool.ranks) != config.worker_count):
        raise NotImplementedError("virtual PP requires untied Qwen3 BF16, TP1/PP>=2 or TP2/PP2, EP-off")
    q_lens, kv_lens, rows = _dense_source_metadata(model, step, input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, hardware_id=pool.hardware_id)
    physical_rows = rows if graph_token_shape is None else graph_token_shape.model_input_tokens
    if graph_token_shape is not None and (model_execution_version != "vllm_0_11_0"
            or graph_token_shape.logical_tokens != rows):
        raise NotImplementedError("pipeline capture requires the declared0.11 stage model program")
    tp = config.tensor_parallel_size
    stages = tuple(tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}")
        for rank in pool.physical_ranks_for_pipeline_stage(stage)) for stage in range(config.pipeline_parallel_size))
    traces = tuple(trace for stage in stages for trace in stage)
    groups = (() if tp == 1 else tuple(ForwardCollectiveGroup(stage,
        namespace=f"{step.step_id}:stage-{index}") for index, stage in enumerate(stages)))
    pipeline = ForwardPipelineGroup(traces, namespace=step.step_id, collective_groups=groups)
    ranges = pipeline_layer_ranges_for_model(model, config.pipeline_parallel_size)

    def run_stage(trace, stage, lane, incoming):
        group = None if tp == 1 else groups[stage]
        if stage > 0:
            # GPUModelRunner.sync_and_slice_intermediate_tensors copies both
            # received roles into worker-owned buffers before model capture.
            # Only active slices are known here; max_num_tokens backing and
            # allocator reservation are not inferred from a scheduled batch.
            prepared = {}
            for name in ("hidden_states", "residual"):
                buffer = trace.tensor("worker.intermediate:" + name,
                    (physical_rows, model.hidden_size), "bf16", persistent=True)
                prepared[name] = SymbolicCopy(trace, "receive:runner-copy:" + name, step.pool_name)(incoming[name], buffer)
            incoming = prepared
            trace.source_scope = {"worker_intermediate_active_window_only": True,
                "worker_intermediate_backing_capacity_bound": False, "actual_provider_bound": False}
        network = Qwen3Model(trace, model, step.pool_name, q_lens, kv_lens, cache_capacity_tokens,
            slot_mapping, layer_range=ranges[stage],
            token_ids=None if physical_rows != rows else input_ids,
            group=group, collective=None if tp == 1 else pool.collective_for("all_reduce"),
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type), hardware_id=pool.hardware_id,
            model_execution_version=model_execution_version, fused_embedding=model_execution_version != "vllm_0_11_0")
        ids = trace.tensor("input_ids", (physical_rows,), "int32") if stage == 0 else None
        pos = trace.tensor("positions", (physical_rows,), "int64")
        if graph_token_shape is not None:
            slots = trace.tensor("capture.slot-mapping", (physical_rows,), "int64")
            for layer in network.layers:
                layer.self_attn.attn.graph_token_shape = graph_token_shape
                layer.self_attn.attn.slot_tensor = slots
            inputs = ({"input_ids": ids} if stage == 0 else incoming)
            trace.begin_capture(key=f"{source_profile}:stage-{stage}:{graph_token_shape.runtime_mode}:{physical_rows}",
                logical_tokens=rows, model_input_tokens=physical_rows,
                inputs={**inputs, "positions": pos, "slot_mapping": slots},
                runtime_mode=graph_token_shape.runtime_mode, eager_kinds=graph_eager_kinds,
                source_stage="vllm_pipeline")
        hidden_states = network(ids, pos, intermediate_tensors=incoming)
        if graph_token_shape is not None:
            trace.end_capture(outputs={"hidden_states": hidden_states} if network.last_stage else hidden_states)
        if network.last_stage:
            indices = trace.tensor("logits_indices", (len(q_lens),), "int32")
            sampled = _plain_greedy_target(trace, model, step.pool_name, group,
                None if tp == 1 else pool.collective_for("all_gather"), hidden_states, indices, q_lens,
                model_execution_version=model_execution_version)
            return None
        boundary = pool.pipeline_p2p_for(stage, stage + 1)
        if tp == 1:
            return {name: pipeline.send(hidden_states[name], config=boundary, pool_name=step.pool_name,
                tensor_name=name, frontier=hidden_states["hidden_states"]) for name in ("hidden_states", "residual")}
        # parallel_state.send_tensor_dict/irecv_tensor_dict: each lane sends
        # reshape(TP,-1)[lane]. Receiver allocates a FULL backing for each
        # role, receives its slice, then replaces the dict value by a fresh
        # gathered tensor. The postprocess closures retain both old backings
        # until recv_tensor_dict has completed both gathers.
        destination = stages[stage + 1][lane]
        result = {}
        for name in ("hidden_states", "residual"):
            tensor = hidden_states[name]
            length = physical_rows * model.hidden_size // tp
            sent = tensor.view_as_shape(-1).narrow(0, lane * length, length)
            full = destination.tensor(f"receive:{stage}:{name}", tensor.shape, tensor.dtype)
            received = full.view_as_shape(-1).narrow(0, lane * length, length)
            wire = pipeline.send(sent, config=boundary, pool_name=step.pool_name,
                tensor_name=name, frontier=hidden_states["hidden_states"], out=received)
            if model_execution_version == "vllm_0_11_0":
                # 0.11 recv_tensor_dict receives/gathers one role before
                # allocating the next. Its slice's old full backing dies
                # after this fresh gather, unlike0.30 postprocess closures.
                result[name] = groups[stage + 1].call("receive:" + name, wire,
                    pool_name=step.pool_name, config=pool.collective_for("all_gather"), inplace=False, dim=0,
                    output_shape=(physical_rows, model.hidden_size))
                del wire, received, full
            else:
                result[name] = wire
        return result

    def gather_incoming(trace, stage, wire):
        return {name: groups[stage].call("receive:" + name, wire[name], pool_name=step.pool_name,
            config=pool.collective_for("all_gather"), inplace=False, dim=0,
            output_shape=(physical_rows, model.hidden_size)) for name in ("hidden_states", "residual")}

    incoming = [None] * tp
    for stage, stage_traces in enumerate(stages):
        next_incoming = []
        for lane, trace in enumerate(stage_traces):
            if tp > 1 and stage > 0 and model_execution_version == "vllm_0_30_0":
                gathered = gather_incoming(trace, stage, incoming[lane])
                incoming[lane] = None
            else:
                gathered = incoming[lane]
                if tp > 1 and stage > 0:
                    incoming[lane] = None
            next_incoming.append(run_stage(trace, stage, lane, gathered))
            del gathered
        incoming = next_incoming
    if graph_token_shape is not None:
        for group in groups:
            _remap_collective_capture_nodes(group)
    return pipeline


def trace_dense_parallel_forward(model, step, pool, *, input_ids, positions,
                                 slot_mapping, cache_capacity_tokens, source_profile, graph_token_shape=None,
                                 graph_eager_kinds=(), model_execution_version="vllm_0_30_0", sample_logits=True,
                                 cast_logits=False, graph_inputs=None):
    from .graph_padding import VLLMDenseGraphInputs
    config = pool.parallel_plan.config
    if (pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
            or len(pool.ranks) < 2 or len(pool.ranks) != config.tensor_parallel_size
            or config.enable_expert_parallel
            or any(value != 1 for value in (config.pipeline_parallel_size,
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("parallel dense source requires TP>=2, PP/SP/CP/DP1 and EP-off")
    if model.model_type == "qwen3":
        _validate_qwen3_source(model)
        model_type = Qwen3Model
    elif model.model_type == "llama":
        _validate_llama_source(model)
        model_type = LlamaModel
    else:
        raise NotImplementedError("parallel dense source requires an adapted Qwen3 or Llama model")
    if graph_inputs is not None:
        if not isinstance(graph_inputs, VLLMDenseGraphInputs):
            raise TypeError("dense capture requires its typed retained input state")
        if (graph_token_shape is None or model_execution_version != "vllm_0_30_0"
                or graph_inputs.request_ids != tuple(request.request_id for request in step.requests)):
            raise ValueError("dense retained input state must match this captured vLLM0.30 request order")
        if (len(graph_inputs.retained_padding_token_ids) != graph_token_shape.padding_tokens
                or any(token >= model.vocab_size for token in graph_inputs.retained_padding_token_ids)):
            raise ValueError("dense retained padding token IDs must cover exactly the capture tail within the vocabulary")
    if graph_token_shape is not None:
        if pool.hardware_id == "ascend_910b" and model_execution_version != "vllm_0_18_0":
            raise NotImplementedError("Ascend captured model requires its explicit0.18 source")
        if len(pool.ranks) != 2 or pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b"):
            raise NotImplementedError("parallel source capture requires its adapted dense TP2 source")
        masked_capture = (model_execution_version == "vllm_0_11_0"
            or (model_execution_version == "vllm_0_24_0" and pool.hardware_id == "metax_c500")
            or (model_execution_version == "vllm_0_18_0" and pool.hardware_id == "ascend_910b"))
        if (graph_token_shape.model_input_tokens != graph_token_shape.logical_tokens
                and not masked_capture and graph_inputs is None):
            raise NotImplementedError("TP2 captured embedding requires retained padding token IDs; only full buckets are bound")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    for trace in traces:
        _trace_dense_forward(model, step, input_ids=input_ids, positions=positions,
            slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
            source_profile=source_profile, rank=trace.rank, namespace=trace.namespace,
            model_type=model_type, trace=trace, group=group,
            collective=pool.collective_for("all_reduce"), gather=pool.collective_for("all_gather"),
            fused_embedding=(model_execution_version != "vllm_0_11_0" and pool.hardware_id not in ("metax_c500", "ascend_910b")),
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type), hardware_id=pool.hardware_id,
            graph_token_shape=graph_token_shape, graph_eager_kinds=graph_eager_kinds,
            model_execution_version=model_execution_version, sample_logits=sample_logits, cast_logits=cast_logits,
            retained_padding_token_ids=None if graph_inputs is None else graph_inputs.retained_padding_token_ids)
        if graph_inputs is not None:
            capture = trace.source_capture
            capture["padding_ids"] = "caller_declared_retained_source_values"
            capture["padding_positions"] = 0
            capture["retained_padding_inputs"] = {
                "request_ids": graph_inputs.request_ids, "source_kind": graph_inputs.source_kind,
                "source_id": graph_inputs.source_id, "token_ids": graph_inputs.retained_padding_token_ids}
    if graph_token_shape is not None:
        _remap_collective_capture_nodes(group)
    return group



def _dense_parallel_source_prefix(model, step, pool, arguments, *, mlp_factories=None, tail_forward=None,
                                  binding_target=False, binding_tail_nodes=None):
    """Execute the actual first model pipeline range on both source ranks."""
    q_lens, kv_lens, rows = _dense_source_metadata(model, step,
        input_ids=arguments["input_ids"], positions=arguments["positions"],
        slot_mapping=arguments["slot_mapping"], cache_capacity_tokens=arguments["cache_capacity_tokens"],
        source_profile=arguments["source_profile"], hardware_id=pool.hardware_id, allow_unknown_input_ids=True)
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    shape = arguments["graph_token_shape"]
    for trace in traces:
        network = Qwen3Model(trace, model, step.pool_name, q_lens, kv_lens,
            arguments["cache_capacity_tokens"], arguments["slot_mapping"],
            group=group, collective=pool.collective_for("all_reduce"),
            token_ids=None if shape.padding_tokens else arguments["input_ids"],
            fused_embedding=False, layer_range=(0, 1), hardware_id=pool.hardware_id, mlp_factories=mlp_factories,
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type),
            model_execution_version=arguments["model_execution_version"])
        ids = trace.tensor("input_ids", (shape.model_input_tokens,), "int32")
        positions = trace.tensor("positions", (shape.model_input_tokens,), "int64")
        slots = trace.tensor("capture.slot-mapping", (shape.model_input_tokens,), "int64")
        for layer in network.layers:
            layer.self_attn.attn.graph_token_shape = shape
            layer.self_attn.attn.slot_tensor = slots
            if arguments["model_execution_version"] == "vllm_0_18_0":
                layer.self_attn.attn.graph_attention_source = "ascend_0_18_full_fia"
        if binding_target:
            indices = tuple(sum(q_lens[:index + 1]) - 1 for index in range(len(q_lens)))
            logits_indices = trace.tensor("logits_indices", (len(indices),), "int32")
        trace.begin_capture(key=f"{arguments['source_profile']}:{shape.runtime_mode}:{shape.model_input_tokens}",
            logical_tokens=rows, model_input_tokens=shape.model_input_tokens,
            inputs={"input_ids": ids, "positions": positions, "slot_mapping": slots},
            runtime_mode=shape.runtime_mode, eager_kinds=arguments["graph_eager_kinds"])
        output = network(ids, positions)
        if binding_target:
            hidden = (output["hidden_states"] if tail_forward is None else
                tail_forward(trace, group, output["hidden_states"]))
            capture_output, _ = _dense_final_norm(trace, model, pool.name, pool.hardware_id)(
                hidden, output["residual"])
        else:
            capture_output = (output["hidden_states"] if tail_forward is None else
                tail_forward(trace, group, output["hidden_states"]))
        trace.end_capture(capture_output)
        if binding_target:
            before = len(trace._nodes)
            selected, logits, sampled = _dense_forward_target(trace, model, pool.name, capture_output, logits_indices, indices,
                group=group, gather=pool.collective_for("all_gather"),
                model_execution_version=arguments["model_execution_version"], hardware_id=pool.hardware_id,
                sample_logits=arguments["sample_logits"], cast_logits=arguments["cast_logits"], maca_copy_runtime_source=None)
            binding_tail_nodes[trace.rank] = tuple(trace._nodes)[before:]
            del selected, logits, sampled, logits_indices
        # Release source caller references before tracing the next rank. Rank
        # execution order is not a cross-rank storage lifetime order.
        del output, capture_output, network, ids, positions, slots
    _remap_collective_capture_nodes(group)
    return group


def _dense_parallel_metadata_prefix(model, step, pool, arguments, program):
    """Bind the recorded actual two-rank prefix with shared GQA metadata."""
    from copy import copy
    result = copy(program.prefix)
    result.traces = tuple(_dense_metadata_prefix(model, step,
        {**arguments, "hardware_id": pool.hardware_id}, rank_program,
        tp_size=pool.parallel_plan.config.tensor_parallel_size) for rank_program in program.rank_programs)
    return result


def _dense_dynamic_bindings(source, prefix, tail_nodes, model, *, node_bindings=(), collective_attribute_bindings=()):
    nodes, persistent, transient, views = list(node_bindings), {}, {}, {}
    collective_geometry = []
    dynamic_shapes = {"rows", "elements", "m"}
    for full, witness in zip(source.traces, prefix.traces, strict=True):
        pairs = []
        for layer in range(model.num_hidden_layers):
            target, origin = full.qualify(f"layer-{layer}"), witness.qualify("layer-0")
            attention = witness._nodes[origin + ":attention"]
            piecewise = witness.source_captures[0]["runtime_mode"] == "PIECEWISE"
            attention_shapes = (("batch", "q_len", "kv_len")
                if piecewise or ("mathematical_program" in attention.attributes and attention.attributes["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1")
                else ("kv_len",))
            attention_attributes = (("q_lens", "kv_lens", "source_output_rows") if piecewise
                else ("q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens"))
            nodes.append((target + ":attention", origin + ":attention", attention_shapes,
                attention_attributes))
            nodes.append((target + ":attention:kv-update", origin + ":attention:kv-update", ("elements", "seq_len"),
                ("symbolic_slot_mapping",)))
            pairs.extend(((target + ":attention", origin + ":attention"), (target + ":attention:kv-update", origin + ":attention:kv-update")))
        for name in tail_nodes[witness.rank]:
            if name in witness._collective_ids:
                continue
            node = witness._nodes[name]
            nodes.append((name, name, tuple(key for key in node.shape if key in dynamic_shapes),
                ("selected_indices",) if "selected_indices" in node.attributes else ()))
            pairs.append((name, name))
        for identifier, contract in prefix._contracts.items():
            if contract.attributes["operation"] != "all_gather":
                continue
            if full.rank == source.participants[0]:
                collective_geometry.append((identifier, identifier))
            pairs.append((source._submissions[identifier][full.rank], prefix._submissions[identifier][witness.rank]))
        roots = {}
        for target, origin in pairs:
            for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs"):
                left, right = full._nodes[target].attributes[field], witness._nodes[origin].attributes[field]
                if len(left) != len(right):
                    return None
                for target_view, origin_view in zip(left, right, strict=True):
                    target_root, origin_root = full._views[target_view].allocation_id, witness._views[origin_view].allocation_id
                    if target_root in roots and roots[target_root] != origin_root:
                        return None
                    roots[target_root] = origin_root
                    views[target_view] = origin_view
        for target_root, origin_root in roots.items():
            left = tuple(view for view in full._views.values() if view.allocation_id == target_root)
            right = tuple(view for view in witness._views.values() if view.allocation_id == origin_root)
            target = persistent if target_root in full._persistent_ids else transient
            target[target_root] = origin_root
            if len(left) != len(right):
                # Shared metadata roots may feed one narrow view per layer;
                # their actual ports above give every occurrence explicitly.
                views[target_root] = origin_root
                continue
            for old, current in zip(left, right, strict=True):
                if (old.shape, old.strides_bytes, old.offset_bytes, old.dtype) != (
                        current.shape, current.strides_bytes, current.offset_bytes, current.dtype):
                    return None
                views[old.view_id] = current.view_id
    return ForwardCollectiveSourceProgram(source, prefix, node_bindings=nodes,
        allocation_bindings=tuple(persistent.items()), transient_allocation_bindings=tuple(transient.items()),
        view_bindings=tuple(views.items()), capture_bindings=((0, 0, ("logical_tokens",)),),
        collective_geometry_bindings=tuple(collective_geometry),
        collective_attribute_bindings=collective_attribute_bindings)


def _dense_parallel_dynamic_program(model, step, pool, arguments, *, cache, launch_policy):
    if not _DENSE_METADATA_SOURCE_CONTRACT.unchanged():
        source = trace_dense_parallel_forward(model, step, pool, **arguments)
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
        return source
    shape = arguments["graph_token_shape"]
    # Logical rows remain typed bindings; runtime control flow, the physical
    # bucket and all static source facts stay in the key.
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens", "graph_token_shape"}
    piecewise = shape.runtime_mode == "PIECEWISE"
    domain = "dense_dynamic_piecewise" if piecewise else "dense_dynamic_q1"
    key = (domain, model, pool.name, pool.hardware_id, pool.ranks, pool.parallel_plan.config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        shape.runtime_mode, shape.model_input_tokens,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    if piecewise:
        key += (("source_control_flow", shape.logical_tokens < shape.model_input_tokens,
            all(request.scheduled_tokens == 1 for request in step.requests)),)
    if arguments["model_execution_version"] == "vllm_0_18_0":
        # FIA's direct K/V and persistent cache reads are different programs.
        key += (("ascend_fia_prefill_no_cache", all(request.computed_tokens == 0 for request in step.requests)),)
    program = cache.lookup(key)
    if piecewise and program is not None and ("_dense_parallel_metadata_guard" not in program.__dict__
            or program._dense_parallel_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        raise ValueError("cached dense piecewise source lost its source metadata contract")
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"])
        + ":tp2-geometry" if program is None else program.source.namespace)
    current = replace(step, step_id=namespace)
    tail_nodes = {}
    prefix = _dense_parallel_source_prefix(model, current, pool, arguments,
        binding_target=True, binding_tail_nodes=tail_nodes)
    for trace in prefix.traces:
        trace.bind_launch_policy(launch_policy)
    if program is not None:
        result = program.bind(prefix)
        if result is not None:
            return result
        if piecewise:
            raise ValueError("cached dense piecewise source rejected its declared control-flow and geometry key")
    source = trace_dense_parallel_forward(model, current, pool, **arguments)
    for trace in source.traces:
        trace.bind_launch_policy(launch_policy)
    program = _dense_dynamic_bindings(source, prefix, tail_nodes, model)
    if program is None:
        return source
    program._dense_parallel_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:
        raise RuntimeError("recorded typed geometry rejected its own actual prefix")
    return result


def _dense_parallel_eager_domain(model, step, pool, arguments):
    config = pool.parallel_plan.config
    return (model.model_type == "qwen3" and model.num_hidden_layers == 64 and model.hidden_size == 5120
        and not model.num_experts and model.compute_dtype == model.weight_dtype == "bf16"
        and (pool.hardware_id, arguments["model_execution_version"]) in (
            ("nvidia_a800", "vllm_0_11_0"), ("nvidia_h200", "vllm_0_11_0"),
            ("metax_c500", "vllm_0_24_0"), ("ascend_910b", "vllm_0_18_0"))
        and (pool.hardware_id != "ascend_910b"
            or any(request.scheduled_tokens != 1 for request in step.requests)
            or all(request.computed_tokens == 0 for request in step.requests))
        and len(pool.ranks) == config.tensor_parallel_size == 2
        and not config.enable_expert_parallel and config.pipeline_parallel_size == 1 and config.is_rank_local
        and arguments["graph_token_shape"] is None
        and not arguments["graph_eager_kinds"] and arguments["graph_inputs"] is None
        and arguments["input_ids"] is None and arguments["sample_logits"] and not arguments["cast_logits"]
        and all(request.mtp_proposer_mode == "disabled" for request in step.requests))


def _dense_parallel_eager_prefix(model, step, pool, arguments):
    q_lens, kv_lens, rows = _dense_source_metadata(model, step,
        input_ids=arguments["input_ids"], positions=arguments["positions"], slot_mapping=arguments["slot_mapping"],
        cache_capacity_tokens=arguments["cache_capacity_tokens"], source_profile=arguments["source_profile"],
        hardware_id=pool.hardware_id, allow_unknown_input_ids=True)
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    for trace in traces:
        network = Qwen3Model(trace, model, pool.name, q_lens, kv_lens,
            arguments["cache_capacity_tokens"], arguments["slot_mapping"], group=group,
            collective=pool.collective_for("all_reduce"), token_ids=arguments["input_ids"],
            fused_embedding=False, layer_range=(0, 2), hardware_id=pool.hardware_id,
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type),
            model_execution_version=arguments["model_execution_version"])
        ids = trace.tensor("input_ids", (rows,), "int32")
        pos = trace.tensor("positions", (rows,), "int64")
        indices = tuple(sum(q_lens[:i + 1]) - 1 for i in range(len(q_lens)))
        logits_indices = trace.tensor("logits_indices", (len(indices),), "int32")
        intermediate = network(ids, pos)
        hidden, residual = _dense_final_norm(trace, model, pool.name, pool.hardware_id)(
            intermediate["hidden_states"], intermediate["residual"])
        selected, logits, output = _dense_forward_target(trace, model, pool.name, hidden, logits_indices, indices,
            group=group, gather=pool.collective_for("all_gather"), model_execution_version=arguments["model_execution_version"],
            hardware_id=pool.hardware_id, sample_logits=arguments["sample_logits"], cast_logits=arguments["cast_logits"],
            maca_copy_runtime_source=None)
        del selected, logits, output, hidden, residual, intermediate, logits_indices, ids, pos, network
    return group


def _dense_eager_origin_suffix(suffix):
    pieces = suffix.split(":")
    return ":".join("layer-0" if piece == "layer-0" else "layer-1"
        if piece.startswith("layer-") and piece[6:].isdigit() else piece for piece in pieces)


def _dense_eager_bindings(source, prefix, model):
    nodes, persistent, transient, views, collectives = [], {}, {}, {}, []
    dynamic_shape_fields = {"rows", "elements", "m", "q_len", "kv_len", "seq_len", "batch", "tok"}
    dynamic_attributes = {"q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens",
        "symbolic_slot_mapping", "selected_indices", "index_count"}
    # Eager attention returns the actual row extent. Grouped Q/K norm
    # materialization likewise takes its outer groups from current T.
    dynamic_attributes.update(("source_output_rows", "input_outer_groups"))
    for full, witness in zip(source.traces, prefix.traces, strict=True):
        pairs = []
        for name, node in full._nodes.items():
            suffix = _dense_eager_origin_suffix(name[len(full.namespace) + 1:])
            origin = witness.qualify(suffix)
            if origin not in witness._nodes:
                return None
            current = witness._nodes[origin]
            if (node.kind, node.dtype, node.variant, node.rank) != (current.kind, current.dtype, current.variant, current.rank):
                return None
            if name not in full._collective_ids:
                fields = dynamic_shape_fields | ({"source_rows"} if "selected_indices" in node.attributes else set())
                nodes.append((name, origin, tuple(key for key in node.shape if key in fields),
                    tuple(key for key in node.attributes if key in dynamic_attributes)))
            pairs.append((name, origin))
        for target, origin in pairs:
            for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs"):
                left, right = full._nodes[target].attributes[field], witness._nodes[origin].attributes[field]
                if len(left) != len(right):
                    return None
                for target_view, origin_view in zip(left, right, strict=True):
                    old, current = full._views[target_view], witness._views[origin_view]
                    if (old.shape, old.strides_bytes, old.offset_bytes, old.dtype) != (
                            current.shape, current.strides_bytes, current.offset_bytes, current.dtype):
                        return None
                    if target_view not in views:
                        views[target_view] = origin_view
                    mapping = persistent if old.allocation_id in full._persistent_ids else transient
                    if old.allocation_id not in mapping:
                        mapping[old.allocation_id] = current.allocation_id
        for mapping in (persistent, transient):
            for target, origin in tuple(mapping.items()):
                if target not in full._allocations:
                    continue
                views[target] = origin
                left = tuple(view for view in full._views.values() if view.allocation_id == target)
                right = tuple(view for view in witness._views.values() if view.allocation_id == origin)
                if len(left) == len(right):
                    for old, current in zip(left, right, strict=True):
                        if old.view_id not in views:
                            views[old.view_id] = current.view_id
    for target, contract in source._contracts.items():
        suffix = _dense_eager_origin_suffix(target[len(source.namespace) + 1:])
        collectives.append((target, prefix.namespace + ":" + suffix))
    return ForwardCollectiveSourceProgram(source, prefix, node_bindings=nodes,
        allocation_bindings=tuple(persistent.items()), transient_allocation_bindings=tuple(transient.items()),
        view_bindings=tuple(views.items()), collective_geometry_bindings=tuple(collectives))


def _dense_parallel_eager_program(model, step, pool, arguments, *, cache, launch_policy):
    if not _DENSE_METADATA_SOURCE_CONTRACT.unchanged():
        source = trace_dense_parallel_forward(model, step, pool, **arguments)
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
        return source
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens"}
    key = ("dense_eager_geometry", model, pool.name, pool.hardware_id, pool.ranks, pool.parallel_plan.config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    # Attention's all-Q1 program and TP logits' one-row view/reorder are
    # distinct source branches, independent of variable eager geometry.
    key += (("source_control_flow", all(request.scheduled_tokens == 1 for request in step.requests),
        len(step.requests) == 1),)
    if arguments["model_execution_version"] == "vllm_0_18_0":
        # Fresh eager FIA reads contiguous K/V; mixed cached FIA reads pages.
        # Pure cached Q1 may select PA and is outside this reuse domain.
        key += (("ascend_fia_prefill_no_cache", all(request.computed_tokens == 0 for request in step.requests)),)
    program = cache.lookup(key)
    if program is not None and (
            "_dense_eager_metadata_guard" not in program.__dict__
            or program._dense_eager_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        raise ValueError("cached dense eager source lost its source metadata contract")
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"])
        + ":tp2-eager-geometry" if program is None else program.source.namespace)
    current = replace(step, step_id=namespace)
    prefix = _dense_parallel_eager_prefix(model, current, pool, arguments)
    for trace in prefix.traces:trace.bind_launch_policy(launch_policy)
    if program is not None:
        result = program.bind(prefix)
        if result is not None:return result
        raise ValueError("cached dense eager source rejected its declared control-flow and geometry key")
    source = trace_dense_parallel_forward(model, current, pool, **arguments)
    for trace in source.traces:trace.bind_launch_policy(launch_policy)
    program = _dense_eager_bindings(source, prefix, model)
    if program is None:return source
    program._dense_eager_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:raise RuntimeError("eager geometry rejected its actual prefix")
    return result


def _dense_direct_parameters(model, step, pool, arguments, program):
    """Issue the existing native frame from a proved dense FULL recipe."""
    from copy import deepcopy
    from math import prod
    from .traced_forward import (_native_source_scope, _native_recipe_signature,
        _native_recipe_matches, _bind_native_source_parameters)
    shape = arguments["graph_token_shape"]
    if (pool.hardware_id not in ("nvidia_a800", "nvidia_h200")
            or arguments["model_execution_version"] != "vllm_0_11_0"
            or model.compute_dtype != model.weight_dtype or model.compute_dtype != "bf16"
            or shape is None or shape.runtime_mode != "FULL"
            or len(step.requests) != shape.logical_tokens
            or any(request.phase != "decode" or request.scheduled_tokens != 1
                or request.mtp_proposer_mode != "disabled" for request in step.requests)
            or type(program) is not ForwardCollectiveSourceProgram
            or type(program.source) is not ForwardCollectiveGroup
            or type(program.prefix) is not ForwardCollectiveGroup
            or not _native_source_scope(program, program.prefix)
            or program.collective_attribute_bindings or program.collective_geometry_bindings
            or any(type(rank) is not ForwardSourceProgram
                or type(rank.source) is not ForwardTrace or type(rank.prefix) is not ForwardTrace
                or rank.view_bindings or rank.capture_bindings or rank.transient_origins
                for rank in program.rank_programs)
            or any(len(trace.source_captures) != 1
                or trace.source_captures[0]["runtime_mode"] != "FULL"
                or (trace.source_captures[0]["logical_tokens"], trace.source_captures[0]["model_input_tokens"])
                    != (shape.logical_tokens, shape.model_input_tokens)
                for trace in (*program.source.traces, *program.prefix.traces))):
        return None
    if "_dense_parameter_recipe" not in program.__dict__:
        # The ordinary binder owns the first complete source/port/storage proof.
        if program.bind(program.prefix) is None:
            return None
        ranks = []
        for rank in program.rank_programs:
            entries = []
            for target, origin, shapes, attrs in rank.node_bindings:
                node = rank.prefix._nodes[origin]
                if node.kind is WorkloadKind.FLASH_ATTN:
                    if shapes != ("kv_len",) or attrs != ("kv_lens", "model_input_kv_lens"):
                        return None
                    kind = "attention"
                elif node.kind is WorkloadKind.KV_UPDATE:
                    if shapes or attrs != ("symbolic_slot_mapping",):
                        return None
                    kind = "slots"
                else:
                    return None
                entries.append((target, kind))
            origins = tuple(dict.fromkeys(origin for _, origin in rank.allocation_bindings))
            if len(origins) != 1:
                return None
            prefix = rank.prefix
            state = deepcopy((prefix.namespace, prefix.rank, prefix.event_order,
                prefix._persistent_ids, prefix._references, prefix._writers, prefix._readers,
                rank._capture_contract(prefix)))
            ranks.append((tuple(entries), origins[0], tuple(prefix._nodes.items()),
                tuple(prefix._allocations.items()), tuple(prefix._views.items()), state))
        group = program.prefix
        collective_state = deepcopy((group.namespace, group.participants, group.participant_nic_ids,
            group.ring_order, group._explicit_ring_order, group._sequences, group._submissions, group._layouts))
        program._dense_parameter_recipe = (_native_recipe_signature(program), tuple(ranks),
            tuple(group._contracts.items()), collective_state)
    recipe = program._dense_parameter_recipe
    if not _native_recipe_matches(program, recipe[0]):
        return None
    for rank, (_, _, nodes, allocations, views, state) in zip(program.rank_programs, recipe[1], strict=True):
        prefix = rank.prefix
        for actual, expected in ((prefix._nodes, nodes), (prefix._allocations, allocations), (prefix._views, views)):
            if len(actual) != len(expected) or any(name not in actual or actual[name] is not value for name, value in expected):
                return None
        if (prefix.namespace, prefix.rank, prefix.event_order, prefix._persistent_ids,
                prefix._references, prefix._writers, prefix._readers, rank._capture_contract(prefix)) != state:
            return None
    group = program.prefix
    if (len(group._contracts) != len(recipe[2]) or any(name not in group._contracts
            or group._contracts[name] is not node for name, node in recipe[2])
            or (group.namespace, group.participants, group.participant_nic_ids, group.ring_order,
                group._explicit_ring_order, group._sequences, group._submissions, group._layouts) != recipe[3]):
        return None
    metadata = _dense_parameter_metadata(model, step, {**arguments, "hardware_id": pool.hardware_id})
    if type(metadata["kv_len"]) is not int or metadata["kv_len"] < 0:
        raise ValueError("source frame requires canonical bound shape dimensions")
    cache_shape = gqa_cache_shape(arguments["cache_capacity_tokens"],
        model.kv_heads_per_tensor_parallel_rank(pool.parallel_plan.config.tensor_parallel_size), model.head_dim)
    rank_values = []
    for rank, (entries, cache, *_) in zip(program.rank_programs, recipe[1], strict=True):
        updates = {}
        for target, kind in entries:
            if kind == "attention":
                updates[target] = ({"kv_len": metadata["kv_len"]}, {"kv_lens": metadata["kv_lens"],
                    "model_input_kv_lens": metadata["attention_shape"].model_input_kv_lens})
            elif kind == "slots":
                updates[target] = ({}, {"symbolic_slot_mapping": metadata["slot_mapping"]})
            else:
                raise AssertionError("uncompiled dense source parameter kind")
        view = rank.prefix._views[cache]
        allocations = {cache: replace(rank.prefix._allocations[cache], size_bytes=prod(cache_shape) * view.element_bytes)}
        views = {cache: ForwardTrace._dense_view(cache, cache_shape, view.dtype)}
        rank_values.append((updates, allocations, views))
    return _bind_native_source_parameters(program, tuple(rank_values), {})


def _dense_parallel_source_program(model, step, pool, arguments, *, cache, launch_policy):
    """Reuse the recorded Qwen32 BF16 TP2 vendor/version source."""
    config = pool.parallel_plan.config
    shape = arguments["graph_token_shape"]
    mixed = (model.model_type == "qwen3" and model.num_hidden_layers == 64 and model.hidden_size == 5120
        and not model.num_experts and model.compute_dtype == model.weight_dtype == "bf16"
        and (pool.hardware_id, arguments["model_execution_version"]) in (
            ("nvidia_a800", "vllm_0_11_0"), ("nvidia_h200", "vllm_0_11_0"))
        and len(pool.ranks) == config.tensor_parallel_size == 2
        and not config.enable_expert_parallel and config.pipeline_parallel_size == 1 and config.is_rank_local
        and shape is not None and shape.runtime_mode == "PIECEWISE"
        and arguments["graph_eager_kinds"] == ("flash_attn", "kv_update")
        and arguments["graph_inputs"] is None and arguments["sample_logits"] and not arguments["cast_logits"]
        and {request.phase for request in step.requests} == {"prefill", "decode"}
        and all(request.mtp_proposer_mode == "disabled" for request in step.requests))
    if mixed:
        return _dense_parallel_dynamic_program(model, step, pool, arguments,
            cache=cache, launch_policy=launch_policy)
    supported = (model.model_type == "qwen3" and model.num_hidden_layers == 64 and model.hidden_size == 5120
        and not model.num_experts and model.compute_dtype == model.weight_dtype == "bf16"
        and (pool.hardware_id, arguments["model_execution_version"]) in (
            ("nvidia_a800", "vllm_0_11_0"), ("nvidia_h200", "vllm_0_11_0"),
            ("metax_c500", "vllm_0_24_0"), ("ascend_910b", "vllm_0_18_0"))
        and len(pool.ranks) == config.tensor_parallel_size == 2
        and not config.enable_expert_parallel and config.pipeline_parallel_size == 1 and config.is_rank_local
        and shape is not None and shape.runtime_mode == "FULL" and not arguments["graph_eager_kinds"]
        and arguments["graph_inputs"] is None and arguments["sample_logits"] and not arguments["cast_logits"]
        and all(request.phase == "decode" and request.scheduled_tokens == 1
            and request.mtp_proposer_mode == "disabled" for request in step.requests))
    if not supported:
        return (_dense_parallel_eager_program(model, step, pool, arguments, cache=cache, launch_policy=launch_policy)
            if _dense_parallel_eager_domain(model, step, pool, arguments) else
            trace_dense_parallel_forward(model, step, pool, **arguments))
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens"}
    key = (model, pool.name, pool.hardware_id, pool.ranks, config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    if arguments["model_execution_version"] == "vllm_0_18_0":
        # FIA reads direct K/V for a fresh invocation and persistent cache for
        # the other branch. These complete source programs cannot be shared.
        key += (("ascend_fia_prefill_no_cache", all(request.computed_tokens == 0 for request in step.requests)),)
    program = cache.lookup(key)
    metadata_scope = _DENSE_METADATA_SOURCE_CONTRACT.unchanged()
    if not metadata_scope:
        source = trace_dense_parallel_forward(model, step, pool, **arguments)
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
        return source
    if program is not None and (not metadata_scope
            or "_dense_parallel_metadata_guard" not in program.__dict__
            or program._dense_parallel_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        program = None
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"])
        + ":tp2" if program is None else program.source.namespace)
    local_step = replace(step, step_id=namespace)
    if program is not None and metadata_scope:
        bound = _dense_direct_parameters(model, step, pool, arguments, program)
        if bound is not None:
            return bound
    prefix = (_dense_parallel_metadata_prefix(model, step, pool, arguments, program)
        if program is not None and metadata_scope else _dense_parallel_source_prefix(model, local_step, pool, arguments))
    if program is not None:
        bound = program.bind(prefix)
        if bound is not None:
            return bound
    if (metadata_scope and pool.hardware_id in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
            and arguments["input_ids"] is None and 2 <= shape.logical_tokens < shape.model_input_tokens):
        # A previously unseen exact geometry can be proved by the shared
        # physical-bucket program. The unchanged constructor below specializes
        # that complete proof into the ordinary exact-key source program.
        source = _dense_parallel_dynamic_program(model, step, pool, arguments, cache=cache, launch_policy=launch_policy)
        if source.namespace != local_step.step_id:
            local_step = replace(step, step_id=source.namespace)
            prefix = _dense_parallel_source_prefix(model, local_step, pool, arguments)
    else:
        source = trace_dense_parallel_forward(model, local_step, pool, **arguments)
    if launch_policy == "injected_launch":
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
    bindings, allocations = [], []
    for trace, current in zip(source.traces, prefix.traces, strict=True):
        for name, node in current._nodes.items():
            if name not in trace._nodes:
                return source
            actual = trace._nodes[name]
            if ((node.kind, node.pool_name, node.shape, node.dtype, node.variant, node.rank) !=
                    (actual.kind, actual.pool_name, actual.shape, actual.dtype, actual.variant, actual.rank)
                    or {key: value for key, value in node.attributes.items() if key != "launch_policy"} !=
                    {key: value for key, value in actual.attributes.items() if key != "launch_policy"}):
                return source
        for layer in range(model.num_hidden_layers):
            target, origin = trace.qualify(f"layer-{layer}"), current.qualify("layer-0")
            bindings.extend(((target + ":attention", origin + ":attention", ("kv_len",),
                ("kv_lens", "model_input_kv_lens")),
                (target + ":attention:kv-update", origin + ":attention:kv-update", (), ("symbolic_slot_mapping",))))
            allocations.append((target + ":kv-cache", origin + ":kv-cache"))
    program = ForwardCollectiveSourceProgram(source, prefix, node_bindings=bindings, allocation_bindings=allocations)
    if metadata_scope:
        program._dense_parallel_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:
        raise RuntimeError("new collective program rejected its own actual source prefix")
    return result


def _moe_parallel_source_prefix(model, step, pool, routing, arguments, template, *, binding_target=False, binding_tail_nodes=None):
    """Actual first layer plus current expert source leaves for all later layers."""
    from functools import partial
    shape = arguments["graph_token_shape"]
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, routing, invocations, shape.model_input_tokens)
    factories = tuple(partial(Qwen3MoeSparseMoeBlock, resolution=resolution, invocation_id=invocation,
        ep_size=pool.parallel_plan.config.expert_parallel_size,
        model_execution_version=arguments["model_execution_version"])
        for resolution, invocation in zip(resolutions, invocations, strict=True))

    def tail(trace, group, output):
        original = next(current for current in template.traces if current.rank == trace.rank)
        roots, ports = {}, {}

        def local(name):
            if not name.startswith(trace.namespace + ":"):
                raise ValueError("expert source port belongs to another source program")
            return name[len(trace.namespace) + 1:]

        def port(name):
            if name in ports:
                return ports[name]
            view = original._views[name]
            storage = view.allocation_id
            if storage not in roots:
                root = original._views[storage]
                roots[storage] = trace.tensor(local(storage), root.shape, root.dtype,
                    persistent=storage in original._persistent_ids,
                    memory_domain=original._allocations[storage].memory_domain)
            result = roots[storage] if name == storage else trace.alias(roots[storage], local(name),
                shape=view.shape, strides_bytes=view.strides_bytes, offset_bytes=view.offset_bytes, dtype=view.dtype)
            if (result.view.shape, result.view.strides_bytes, result.view.offset_bytes, result.view.dtype) != (
                    view.shape, view.strides_bytes, view.offset_bytes, view.dtype):
                raise ValueError("expert source input geometry changed from the recorded source")
            ports[name] = result
            return result

        for node in original._nodes.values():
            if node.kind is not WorkloadKind.FUSED_MOE:
                continue
            layer = node.attributes["routing_provenance"]["layer_index"]
            if layer == 0:
                continue  # Already executed by the actual first-layer model.
            names = node.attributes["symbolic_tensor_inputs"]
            expected_roles = ("hidden_states", "routing_weights", "expert_ids", "w13_weight", "w2_weight")
            if (node.attributes["memory_input_roles"] != expected_roles or len(names) != len(expected_roles)
                    or len(set(names)) != len(names)
                    or node.attributes["memory_output_roles"] != ("output",)
                    or node.attributes["expert_parameter_layout"] != "unquantized_checkpoint_expert_major_nk"):
                raise ValueError("expert replay requires the exact declared public input/parameter roles")
            x, weights, ids = (port(names[index]) for index in range(3))
            inplace = original._views[node.attributes["symbolic_tensor_outputs"][0]].allocation_id == x.view.allocation_id
            expert = SymbolicUnquantizedExperts(trace, local(node.node_id), pool.name, model,
                tp_size=pool.parallel_plan.config.tensor_parallel_size, resolution=resolutions[layer],
                invocation_id=invocations[layer], expert_parallel_size=pool.parallel_plan.config.expert_parallel_size,
                expert_rank=node.attributes["expert_rank"], inplace=inplace,
                routing_weight_dtype=node.attributes["routing_weight_dtype"],
                model_execution_version=node.attributes["model_execution_version"])
            output = expert(x, weights, ids)
            name = f"layer-{layer}:all-reduce"
            contract = template._contracts[template.namespace + ":" + name]
            output = group.call(name, output, pool_name=pool.name, config=pool.collective_for("all_reduce"),
                inplace=contract.attributes["symbolic_collective_storage"] == "inplace",
                moe_invocation_id=invocations[layer])
        return output

    return _dense_parallel_source_prefix(model, replace(step, step_id=template.namespace), pool, arguments,
        mlp_factories=factories, tail_forward=tail,
        binding_target=binding_target, binding_tail_nodes=binding_tail_nodes)


def _moe_exact_source_bindings(source, prefix, model):
    bindings, allocations, collectives = [], [], []
    expert_fields = ("tokens_per_expert", "local_tokens_per_expert", "moe_invocation_id", "routing_provenance")
    for trace, current in zip(source.traces, prefix.traces, strict=True):
        for layer in range(model.num_hidden_layers):
            target, origin = trace.qualify(f"layer-{layer}"), current.qualify("layer-0")
            attention_attributes = ("kv_lens", "model_input_kv_lens")
            bindings.extend(((target + ":attention", origin + ":attention", ("kv_len",), attention_attributes),
                (target + ":attention:kv-update", origin + ":attention:kv-update", (), ("symbolic_slot_mapping",))))
            allocations.append((target + ":kv-cache", origin + ":kv-cache"))
        for name, node in current._nodes.items():
            if node.kind is WorkloadKind.FUSED_MOE:
                bindings.append((name, name, (), expert_fields))
        if trace.rank == source.participants[0]:
            collectives.extend((identifier, identifier, ("moe_invocation_id",))
                for identifier, node in prefix._contracts.items() if "moe_invocation_id" in node.attributes)
    return ForwardCollectiveSourceProgram(source, prefix, node_bindings=bindings,
        allocation_bindings=allocations, collective_attribute_bindings=collectives)


def _moe_dynamic_bindings(source, prefix, tail_nodes, model):
    expert_fields = ("tokens_per_expert", "local_tokens_per_expert", "moe_invocation_id", "routing_provenance")
    nodes = tuple((name, name, (), expert_fields)
        for trace in prefix.traces for name, node in trace._nodes.items()
        if node.kind is WorkloadKind.FUSED_MOE)
    collectives = tuple((name, name, ("moe_invocation_id",))
        for name, node in prefix._contracts.items() if "moe_invocation_id" in node.attributes)
    return _dense_dynamic_bindings(source, prefix, tail_nodes, model,
        node_bindings=nodes, collective_attribute_bindings=collectives)


def _moe_parallel_dynamic_program(model, step, pool, routing, arguments, *, cache, launch_policy, exact_key):
    """Prove each new logical extent with current source operations and ports."""
    shape = arguments["graph_token_shape"]
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, routing, invocations, shape.model_input_tokens)
    # Preserve the actual per-layer/per-rank source branch, even when a
    # later routing realization leaves a rank with no local assignments.
    branch = tuple(tuple(any(unquantized_expert_source_metadata(model, resolution, invocation,
        rows=shape.model_input_tokens, expert_parallel_size=pool.parallel_plan.config.expert_parallel_size,
        expert_rank=rank)["local_tokens_per_expert"]) for rank in range(len(pool.ranks)))
        for resolution, invocation in zip(resolutions, invocations, strict=True))
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens", "graph_token_shape"}
    domain = "moe_dynamic_q1" if shape.runtime_mode == "FULL" else "moe_dynamic_piecewise"
    key = (domain, model, pool.name, pool.hardware_id, pool.ranks, pool.parallel_plan.config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        routing.policy_descriptor_sha256, shape.runtime_mode, shape.model_input_tokens, branch,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    if shape.runtime_mode == "PIECEWISE":
        key += (("source_control_flow", shape.logical_tokens < shape.model_input_tokens,
            all(request.scheduled_tokens == 1 for request in step.requests)),)
    program = cache.lookup(key)
    if program is not None and ("_moe_metadata_guard" not in program.__dict__
            or program._moe_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        if exact_key is None:
            raise ValueError("cached piecewise source lost its source metadata contract")
        program = None
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"])
        + ":ep2-geometry" if program is None else program.source.namespace)
    if program is None:
        source = trace_qwen3_moe_forward(model, step, pool, routing, source_namespace=namespace, **arguments)
        if launch_policy == "injected_launch":
            for trace in source.traces:
                trace.bind_launch_policy(launch_policy)
    else:
        source = program.source
    tail_nodes = {}
    prefix = _moe_parallel_source_prefix(model, step, pool, routing, arguments, source,
        binding_target=True, binding_tail_nodes=tail_nodes)
    if launch_policy == "injected_launch":
        for trace in prefix.traces:
            trace.bind_launch_policy(launch_policy)
    def specialize(result):
        if exact_key is None:
            return result
        exact = _moe_exact_source_bindings(result, prefix, model)
        exact._moe_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
        cache.store(exact_key, exact)
        return result

    if program is not None:
        result = program.bind(prefix)
        if result is not None:
            return specialize(result)
        if exact_key is None:
            raise ValueError("cached piecewise source rejected its declared control-flow and geometry key")
        # A changed current source contract resumes the ordinary complete
        # source. No cross-domain prediction or cached duration is supplied.
        source = trace_qwen3_moe_forward(model, step, pool, routing, source_namespace=namespace, **arguments)
        if launch_policy == "injected_launch":
            for trace in source.traces:
                trace.bind_launch_policy(launch_policy)
        tail_nodes = {}
        prefix = _moe_parallel_source_prefix(model, step, pool, routing, arguments, source,
            binding_target=True, binding_tail_nodes=tail_nodes)
        if launch_policy == "injected_launch":
            for trace in prefix.traces:
                trace.bind_launch_policy(launch_policy)
    program = _moe_dynamic_bindings(source, prefix, tail_nodes, model)
    if program is None:
        return source
    program._moe_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:
        raise RuntimeError("new MoE geometry rejected its own actual source prefix")
    return specialize(result)



_MOE_PARAMETER_RECIPES = weakref.WeakKeyDictionary()


def _moe_direct_parameters(model, step, pool, routing, arguments, program):
    """Bind actual dynamic parameters directly to the recorded native recipe."""
    from math import prod
    from .traced_forward import (_native_source_scope, _native_recipe_signature,
        _native_recipe_matches, _bind_native_source_parameters)
    shape = arguments["graph_token_shape"]
    if (shape is None or type(shape.logical_tokens) is not int or shape.logical_tokens <= 0
            or type(shape.model_input_tokens) is not int or shape.model_input_tokens < shape.logical_tokens
            or arguments["model_execution_version"] not in ("vllm_0_11_0", "vllm_0_24_0", "vllm_0_18_0")
            or not _native_source_scope(program, program.prefix)):
        return None
    if (shape.runtime_mode != "FULL" or any(request.phase != "decode" or request.scheduled_tokens != 1 for request in step.requests)
            or len(step.requests) != shape.logical_tokens
            or any(len(trace.source_captures) != 1
                or trace.source_captures[0]["runtime_mode"] != "FULL"
                or (trace.source_captures[0]["logical_tokens"],trace.source_captures[0]["model_input_tokens"]) != (shape.logical_tokens,shape.model_input_tokens)
                for trace in (*program.source.traces,*program.prefix.traces))):
        return None
    if program in _MOE_PARAMETER_RECIPES:
        recipe = _MOE_PARAMETER_RECIPES[program]
        if not _native_recipe_matches(program,recipe[0]):
            del _MOE_PARAMETER_RECIPES[program]
    if program not in _MOE_PARAMETER_RECIPES:
        # Compile from the same complete current prefix contracts, once.
        # Unknown or late static edits keep the ordinary source path.
        if program.bind(program.prefix) is None:
            return None
        ranks = []
        for index,rank_program in enumerate(program.rank_programs):
            if rank_program.prefix is not program.prefix.traces[index]:return None
            if rank_program.view_bindings or rank_program.capture_bindings or rank_program.transient_origins:
                return None
            entries = []
            for target, origin, shapes, attrs in rank_program.node_bindings:
                base = rank_program.prefix._nodes[origin]
                if base.kind is WorkloadKind.FLASH_ATTN:
                    if any(name != "kv_len" for name in shapes) or attrs != ("kv_lens","model_input_kv_lens"):
                        return None
                    tag = ("attention",None)
                elif base.kind is WorkloadKind.KV_UPDATE:
                    if shapes or attrs != ("symbolic_slot_mapping",): return None
                    tag = ("slots",None)
                elif base.kind is WorkloadKind.FUSED_MOE:
                    if shapes or attrs != ("tokens_per_expert","local_tokens_per_expert","moe_invocation_id","routing_provenance"):
                        return None
                    tag = ("expert",base.attributes["routing_provenance"]["layer_index"])
                elif base.kind is WorkloadKind.COLLECTIVE:
                    if shapes or attrs != ("moe_invocation_id",):return None
                    invocation = base.attributes["moe_invocation_id"]
                    layers = tuple(node.attributes["routing_provenance"]["layer_index"]
                        for node in rank_program.prefix._nodes.values() if node.kind is WorkloadKind.FUSED_MOE
                        and node.attributes["moe_invocation_id"] == invocation)
                    if len(layers) != 1:return None
                    tag = ("collective",layers[0])
                else:return None
                entries.append((target,origin,shapes,attrs,base,tag))
            allocation_origins = tuple(dict.fromkeys(origin for target,origin in rank_program.allocation_bindings))
            if len(allocation_origins) != 1:return None
            cache, = allocation_origins
            ranks.append((tuple(rank_program.prefix._nodes.items()),tuple(entries),cache))
        joined = []
        for target,origin,attrs in program.collective_attribute_bindings:
            if attrs != ("moe_invocation_id",):return None
            invocation = program.prefix._contracts[origin].attributes["moe_invocation_id"]
            layers = tuple(node.attributes["routing_provenance"]["layer_index"]
                for node in program.prefix.traces[0]._nodes.values() if node.kind is WorkloadKind.FUSED_MOE
                and node.attributes["moe_invocation_id"] == invocation)
            if len(layers) != 1:return None
            joined.append((target,layers[0]))
        _MOE_PARAMETER_RECIPES[program] = (_native_recipe_signature(program),tuple(ranks),
            tuple(program.prefix._contracts.items()),tuple(joined))
    recipe = _MOE_PARAMETER_RECIPES[program]
    for rank_program,(expected,entries,cache) in zip(program.rank_programs,recipe[1],strict=True):
        if (len(rank_program.prefix._nodes) != len(expected)
                or any(name not in rank_program.prefix._nodes or rank_program.prefix._nodes[name] is not node
                    for name,node in expected)):
            return None
    if (len(program.prefix._contracts) != len(recipe[2]) or any(name not in program.prefix._contracts
            or program.prefix._contracts[name] is not node for name,node in recipe[2])):
        return None
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, routing, invocations, shape.model_input_tokens)
    metadata = _dense_parameter_metadata(model,step,{**arguments,"hardware_id":pool.hardware_id})
    if type(metadata["kv_len"]) is not int or metadata["kv_len"] < 0:
        raise ValueError("source frame requires canonical bound shape dimensions")
    cache_shape = gqa_cache_shape(arguments["cache_capacity_tokens"],
        model.kv_heads_per_tensor_parallel_rank(pool.parallel_plan.config.tensor_parallel_size),model.head_dim)
    placement = {};rank_values = []
    for rank_program,(expected,entries,cache) in zip(program.rank_programs,recipe[1],strict=True):
        updates = {}
        for target,origin,shapes,attrs,base,(kind,layer) in entries:
            if kind == "attention":
                updates[target] = ({name:metadata[name] for name in shapes},
                    {"kv_lens":metadata["kv_lens"],"model_input_kv_lens":metadata["attention_shape"].model_input_kv_lens})
            elif kind == "slots":
                updates[target] = ({},{"symbolic_slot_mapping":metadata["slot_mapping"]})
            elif kind == "collective":
                updates[target] = ({},{"moe_invocation_id":invocations[layer]})
            elif kind == "expert":
                resolution = resolutions[layer]
                counts = resolution.tokens_per_expert
                key = (id(counts),base.shape["tokens"],base.attributes["expert_parallel_size"],base.attributes["expert_rank"])
                if key not in placement:
                    placement[key] = (counts,unquantized_expert_source_metadata(model,resolution,invocations[layer],
                        rows=base.shape["tokens"],expert_parallel_size=base.attributes["expert_parallel_size"],
                        expert_rank=base.attributes["expert_rank"]))
                retained,current = placement[key]
                if retained is not counts:raise RuntimeError("source placement count identity collision")
                if any(current[name] != base.attributes[name] for name in current if name not in attrs):return None
                updates[target] = ({},{"tokens_per_expert":counts,
                    "local_tokens_per_expert":current["local_tokens_per_expert"],
                    "moe_invocation_id":invocations[layer],"routing_provenance":resolution.routing_provenance})
            else:raise AssertionError("uncompiled source parameter kind")
        view = rank_program.prefix._views[cache]
        allocations = {cache:replace(rank_program.prefix._allocations[cache],size_bytes=prod(cache_shape)*view.element_bytes)}
        views = {cache:ForwardTrace._dense_view(cache,cache_shape,view.dtype)}
        rank_values.append((updates,allocations,views))
    collective_values = {target:({},{"moe_invocation_id":invocations[layer]})for target,layer in recipe[3]}
    return _bind_native_source_parameters(program,tuple(rank_values),collective_values)

def _moe_metadata_prefix(model, step, pool, routing, arguments, program):
    """Bind an actual recorded prefix; no source operator/port is synthesized."""
    from copy import copy
    from .runtime_inputs import ExpertRoutingResolution
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, routing, invocations,
        arguments["graph_token_shape"].model_input_tokens)
    result = copy(program.prefix)
    result.traces = tuple(_dense_metadata_prefix(model, step,
        {**arguments, "hardware_id": pool.hardware_id}, rank_program,
        tp_size=pool.parallel_plan.config.tensor_parallel_size) for rank_program in program.rank_programs)
    result._contracts = dict(program.prefix._contracts)
    layers_by_invocation = {}
    dynamic_fields = ("tokens_per_expert", "local_tokens_per_expert", "moe_invocation_id", "routing_provenance")
    placement_by_identity = {}
    native_counts_by_identity = {}
    for trace in result.traces:
        for name, node in tuple(trace._nodes.items()):
            if node.kind is not WorkloadKind.FUSED_MOE:
                continue
            layer = node.attributes["routing_provenance"]["layer_index"]
            layers_by_invocation[node.attributes["moe_invocation_id"]] = layer
            resolution = resolutions[layer]
            native = type(model) is ModelConfig and type(resolution) is ExpertRoutingResolution
            counts = resolution.tokens_per_expert if native else None
            identity = id(counts)
            native = native and type(counts) is tuple
            if native:
                if identity not in native_counts_by_identity:
                    native_counts_by_identity[identity] = (counts, all(type(value) is int for value in counts))
                retained, native = native_counts_by_identity[identity]
                if retained is not counts:
                    raise RuntimeError("source placement count identity collision")
            rows, ep_size, ep_rank = node.shape["tokens"], node.attributes["expert_parallel_size"], node.attributes["expert_rank"]
            native = native and all(type(value) is int for value in (rows, ep_size, ep_rank))
            key = (identity, rows, ep_size, ep_rank)
            if native and key in placement_by_identity:
                # The pure helper's placement depends only on this actual
                # immutable histogram and physical expert partition. Every
                # layer keeps its current invocation and routing ledger.
                current = {**placement_by_identity[key], "moe_invocation_id": invocations[layer],
                    "routing_provenance": resolution.routing_provenance}
            else:
                current = unquantized_expert_source_metadata(model, resolution, invocations[layer],
                    rows=rows, expert_parallel_size=ep_size, expert_rank=ep_rank)
                if native:
                    placement_by_identity[key] = current
            if any(current[key] != node.attributes[key] for key in current if key not in dynamic_fields):
                return None
            trace._nodes[name] = node.with_attributes({**node.attributes,
                **{key: current[key] for key in dynamic_fields}})
    for identifier, contract in tuple(result._contracts.items()):
        if "moe_invocation_id" not in contract.attributes:
            continue
        invocation = invocations[layers_by_invocation[contract.attributes["moe_invocation_id"]]]
        result._contracts[identifier] = contract.with_attributes({**contract.attributes, "moe_invocation_id": invocation})
        for trace in result.traces:
            local = result._submissions[identifier][trace.rank]
            node = trace._nodes[local]
            trace._nodes[local] = node.with_attributes({**node.attributes, "moe_invocation_id": invocation})
    return result


def _moe_parallel_source_program(model, step, pool, routing, arguments, *, cache, launch_policy):
    """Bind current MoE metadata without changing its real-step routing identity."""
    shape = arguments["graph_token_shape"]
    mixed = (_vllm011_qwen3_moe_ep2_source(model, pool)
        and arguments["model_execution_version"] == "vllm_0_11_0"
        and shape is not None and shape.runtime_mode == "PIECEWISE"
        and arguments["graph_eager_kinds"] == ("flash_attn", "kv_update")
        and arguments["sample_logits"] and not arguments["cast_logits"]
        and routing.source_kind in ("explicit_assumption", "profile_observation")
        and {request.phase for request in step.requests} == {"prefill", "decode"}
        and all(request.mtp_proposer_mode == "disabled" for request in step.requests))
    if mixed and _DENSE_METADATA_SOURCE_CONTRACT.unchanged():
        return _moe_parallel_dynamic_program(model, step, pool, routing, arguments,
            cache=cache, launch_policy=launch_policy, exact_key=None)
    supported = (_versioned_qwen3_moe_ep2_source(model, pool, arguments["model_execution_version"])
        and shape is not None and shape.runtime_mode == "FULL" and not arguments["graph_eager_kinds"]
        and arguments["sample_logits"] and not arguments["cast_logits"]
        and routing.source_kind in ("explicit_assumption", "profile_observation")
        and all(request.phase == "decode" and request.scheduled_tokens == 1
            and request.mtp_proposer_mode == "disabled" for request in step.requests))
    if not supported:
        return trace_qwen3_moe_forward(model, step, pool, routing, **arguments)
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens"}
    key = ("moe", model, pool.name, pool.hardware_id, pool.ranks, pool.parallel_plan.config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        routing.policy_descriptor_sha256,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    if arguments["model_execution_version"] == "vllm_0_18_0":
        # FIA direct K/V and persistent-cache readers are distinct source programs.
        key += (("ascend_fia_prefill_no_cache", all(request.computed_tokens == 0 for request in step.requests)),)
    program = cache.lookup(key)
    metadata_domain = (pool.hardware_id, arguments["model_execution_version"]) in (
        ("nvidia_h200", "vllm_0_11_0"), ("nvidia_a800", "vllm_0_11_0"),
        ("metax_c500", "vllm_0_24_0"), ("ascend_910b", "vllm_0_18_0"))
    metadata_scope = metadata_domain and _DENSE_METADATA_SOURCE_CONTRACT.unchanged()
    if program is not None and metadata_domain and (not metadata_scope
            or "_moe_metadata_guard" not in program.__dict__
            or program._moe_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        program = None
    if program is not None:
        if metadata_scope:
            direct = _moe_direct_parameters(model, step, pool, routing, arguments, program)
            if direct is not None:
                return direct
        prefix = (_moe_metadata_prefix(model, step, pool, routing, arguments, program) if metadata_scope
            else _moe_parallel_source_prefix(model, step, pool, routing, arguments, program.source))
        if prefix is None:
            prefix = _moe_parallel_source_prefix(model, step, pool, routing, arguments, program.source)
        if launch_policy == "injected_launch":
            for trace in prefix.traces:
                trace.bind_launch_policy(launch_policy)
        bound = program.bind(prefix)
        if bound is not None:
            return bound
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"])
        + ":ep2" if program is None else program.source.namespace)
    geometry_domain = (metadata_scope and pool.hardware_id in ("nvidia_h200", "nvidia_a800")
        and arguments["model_execution_version"] == "vllm_0_11_0" and model.num_experts == 128
        and model.num_hidden_layers == 48 and model.hidden_size == 2048 and model.moe_intermediate_size == 768
        and arguments["source_profile"] == "cuda_eager_inplace" and arguments["input_ids"] is None
        and routing.source_kind == "profile_observation" and routing.distribution == "model_profile_layer_mean_v1"
        and 2 <= shape.logical_tokens < shape.model_input_tokens)
    if geometry_domain:
        return _moe_parallel_dynamic_program(model, step, pool, routing, arguments,
            cache=cache, launch_policy=launch_policy, exact_key=key)
    source = trace_qwen3_moe_forward(model, step, pool, routing, source_namespace=namespace, **arguments)
    if launch_policy == "injected_launch":
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
    prefix = _moe_parallel_source_prefix(model, step, pool, routing, arguments, source,
        binding_target=arguments["model_execution_version"] == "vllm_0_18_0", binding_tail_nodes={})
    if launch_policy == "injected_launch":
        for trace in prefix.traces:
            trace.bind_launch_policy(launch_policy)
    program = _moe_exact_source_bindings(source, prefix, model)
    if metadata_scope:
        program._moe_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:
        raise RuntimeError("new MoE source program rejected its own actual source prefix")
    return result


def _remap_collective_capture_nodes(group):
    """Use joined collective IDs in every rank's captured source segments."""
    mapping = {local: identifier for identifier, submissions in group._submissions.items()
               for local in submissions.values()}
    for trace in group.traces:
        for capture in trace.source_captures:
            for field in ("node_ids", "model_node_ids", "eager_node_ids"):
                capture[field] = tuple(mapping[node] if node in mapping else node for node in capture[field])
            capture["source_segments"] = tuple(tuple(mapping[node] if node in mapping else node for node in segment)
                for segment in capture["source_segments"])


def _dense_source_metadata(model, step, *, input_ids, positions, slot_mapping,
                           cache_capacity_tokens, source_profile, hardware_id, mtp_source=False,
                           allow_unknown_input_ids=False):
    expected_profile = ("vllm_0_11_glm4_sync" if model.model_type == "glm4_moe"
        else "vllm_ascend_eager" if hardware_id == "ascend_910b" else "cuda_eager_inplace")
    if source_profile != expected_profile:
        raise NotImplementedError("symbolic dense forward requires the declared hardware's eager source profile")
    if not mtp_source and any(r.mtp_proposer_mode != "disabled" or r.mtp_proposed_tokens or r.mtp_proposer_tokens
            or r.mtp_verified_tokens or r.mtp_next_proposed_tokens for r in step.requests):
        raise NotImplementedError("initial symbolic Qwen3 source requires MTP-off")
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    rows = sum(q_lens)
    inputs = (("positions", positions), ("slot_mapping", slot_mapping))
    if input_ids is not None or not allow_unknown_input_ids:
        inputs = (("input_ids", input_ids), *inputs)
    for name, values in inputs:
        if not isinstance(values, tuple) or len(values) != rows or any(type(v) is not int for v in values):
            raise ValueError(f"symbolic {name} must provide one explicit integer per model row")
    if input_ids is not None and any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("symbolic embedding ID outside the checkpoint vocabulary")
    if any(not 0 <= value < model.max_position_embeddings for value in positions):
        raise ValueError("symbolic position outside the checkpoint RoPE cache")
    # gpu_model_runner._prepare_inputs: ordinary-text positions are each
    # request's computed-token count plus its within-query offset.
    expected_positions = tuple(position for request in step.requests
        for position in range(request.computed_tokens, request.computed_tokens + request.scheduled_tokens))
    if positions != expected_positions:
        raise ValueError("symbolic Qwen3 positions disagree with the declared ordinary-text context")
    if type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens):
        raise ValueError("symbolic logical cache capacity cannot hold active sequences")
    if len(set(slot_mapping)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slot_mapping):
        raise ValueError("initial symbolic cache source requires unique valid writable slots")
    return q_lens, kv_lens, rows


def _dense_forward_target(trace, model, pool_name, hidden_states, logits_indices, indices, *,
        group, gather, model_execution_version, hardware_id, sample_logits, cast_logits, maca_copy_runtime_source):
    selected = SymbolicLogitsRowSelect(trace, "logits-boundary-gather", pool_name)(
        hidden_states, logits_indices, selected_indices=indices, model_execution_version=model_execution_version,
        hardware_id=hardware_id)
    if group is None:
        head = SymbolicLinear(trace, "lm-head", pool_name, model.hidden_size,
            model.vocab_size, kind=WorkloadKind.LM_HEAD)
        logits, _ = head(selected)
    else:
        processor = SymbolicTPLogitsProcessor(trace, "lm-head", pool_name,
            hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather, model_execution_version=model_execution_version)
        logits = processor(selected)
    if sample_logits or cast_logits:
        output = SymbolicGreedy(trace, "target-sampler", pool_name)(logits,
            model_execution_version=model_execution_version, sample_logits=sample_logits,
            maca_copy_runtime_source=maca_copy_runtime_source)
    return selected, logits, output if sample_logits or cast_logits else None


def _trace_dense_forward(model, step, *, input_ids, positions, slot_mapping,
                         cache_capacity_tokens, source_profile, rank, namespace, model_type,
                         trace=None, group=None, collective=None, gather=None, mlp_factories=None,
                         fused_embedding=True, rope_program=ROPE_PROGRAM_FUSED, hardware_id="nvidia_h200", ffn_factory=None,
                         graph_token_shape=None, graph_eager_kinds=(), model_execution_version=None, sample_logits=True,
                         cast_logits=False, retained_padding_token_ids=None, af_transport=None,
                         maca_copy_runtime_source=None, ascend_piecewise_source=False):
    """Trace complete Qwen3 target-off/all-greedy source for one scheduled step.

    Positions/slots are explicit caller metadata, never generated token values.
    Ordinary eager TP1 or0.11/A800 dense TP1/TP2 IDs may remain unknown:
    only their INT32 shape port is consumed by the public logical program.
    The0.11 TP1 gather and TP2 masked program also cover captured-buffer tails.
    Persistent KV objects describe a declared logical pool, not a physical
    allocator's normal capacity. Profile assumptions do not bind real runtime.
    """
    model_execution_version = model_type.model_execution_version if model_execution_version is None else model_execution_version
    if type(ascend_piecewise_source) is not bool:
        raise TypeError("Ascend PIECEWISE source selection must be explicit boolean")
    if ascend_piecewise_source and (model_execution_version != "vllm_0_18_0" or hardware_id != "ascend_910b"
            or model_type is not Qwen3Model or model.model_type != "qwen3" or model.compute_dtype != "bf16"
            or group is not None or graph_token_shape is not None and graph_token_shape.runtime_mode != "PIECEWISE"):
        raise NotImplementedError("Ascend PIECEWISE source requires ordinary0.18 Qwen3 BF16 TP1")
    if maca_copy_runtime_source is not None and (
            hardware_id != "metax_c500" or model_execution_version != "vllm_0_24_0"
            or group is not None or model.num_experts or model_type not in (Qwen3Model, LlamaModel)):
        raise NotImplementedError("MACA Copy runtime source requires ordinary C5000.24 dense TP1")
    if type(sample_logits) is not bool or type(cast_logits) is not bool or sample_logits and cast_logits:
        raise TypeError("dense source output scope requires exclusive sampling/cast booleans")
    if cast_logits and model.compute_dtype != "bf16":
        raise NotImplementedError("cast-only sampler source requires BF16 input logits")
    masked_moe = (model_type is Qwen3Model and model.model_type == "qwen3_moe"
        and ((model_execution_version == "vllm_0_11_0" and hardware_id in ("nvidia_a800", "nvidia_h200"))
            or (model_execution_version == "vllm_0_24_0" and hardware_id == "metax_c500")
            or (model_execution_version == "vllm_0_18_0" and hardware_id == "ascend_910b"))
        and mlp_factories is not None and isinstance(group, ForwardCollectiveGroup)
        and len(group.traces) == 2 and not fused_embedding)
    if model_execution_version == "vllm_0_24_0" and (
            hardware_id != "metax_c500" or (model.num_experts and not masked_moe) or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16" or model_type not in (Qwen3Model, LlamaModel)):
        raise NotImplementedError("vLLM0.24 dense model source requires C500 homogeneous BF16 Qwen3 or Llama")
    masked_source = ((model_execution_version == "vllm_0_11_0" and hardware_id in ("nvidia_a800", "nvidia_h200"))
        or (model_execution_version == "vllm_0_24_0" and hardware_id == "metax_c500")
        or (model_execution_version == "vllm_0_18_0" and hardware_id == "ascend_910b"))
    masked_tp2 = (masked_source
        and not fused_embedding and group is not None and len(group.traces) == 2)
    plain_tp1_capture = (masked_source
        and group is None and model_type in (Qwen3Model, LlamaModel) and not model.num_experts)
    static_moe = (model_type is Qwen3Model and model.model_type == "qwen3_moe"
        and mlp_factories is not None and hardware_id == "nvidia_h200")
    static_tp2 = (model_type in (Qwen3Model, LlamaModel) and (not model.num_experts or static_moe)
        and hardware_id == "nvidia_h200" and isinstance(group, ForwardCollectiveGroup)
        and len(group.traces) == 2 and input_ids is not None)
    if not sample_logits and (model_type not in (Qwen3Model, LlamaModel)
            or (not static_moe and (model.num_experts or mlp_factories is not None))
            or hardware_id not in ("nvidia_a800", "nvidia_h200") or source_profile != "cuda_eager_inplace"
            or model_execution_version != "vllm_0_30_0" or (group is not None and not static_tp2) or graph_token_shape is not None
            or ffn_factory is not None):
        raise NotImplementedError("static LM-head-only source requires0.30 CUDA dense TP1/H200 dense TP2 or H200 Qwen3-MoE TP1/validated TP=EP2 eager without MTP/graph")
    q_lens, kv_lens, rows = _dense_source_metadata(model, step, input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, hardware_id=hardware_id,
        allow_unknown_input_ids=(model_type in (Qwen3Model, LlamaModel)
            and (not model.num_experts or masked_moe or (static_moe and model_execution_version == "vllm_0_30_0" and group is None))
            and (group is None or (not fused_embedding and len(group.traces) == 2
                and ((model_execution_version == "vllm_0_11_0" and hardware_id in ("nvidia_a800", "nvidia_h200"))
                    or (model_execution_version in ("vllm_0_24_0", "vllm_0_30_0") and hardware_id == "metax_c500")
                    or (model_execution_version == "vllm_0_18_0" and hardware_id == "ascend_910b"))))
            and (graph_token_shape is None or masked_tp2 or plain_tp1_capture)
            and (mlp_factories is None or static_moe or masked_moe) and ffn_factory is None))
    trace = ForwardTrace(rank=rank, namespace=namespace) if trace is None else trace
    # 0.11 forward_native runs TP1 gather or TP2 mask/gather/fill over ALL
    # captured rows. Logical payload is independent of IDs, including the retained
    # input-buffer tail. Validate scheduled IDs above, but do not invent tail
    # values or pass a logical-length tuple as the physical embedding input.
    embedding_ids = (None if masked_tp2 and graph_token_shape is not None
        and graph_token_shape.padding_tokens else input_ids)
    if retained_padding_token_ids is not None:
        if (graph_token_shape is None or not isinstance(retained_padding_token_ids, tuple)
                or input_ids is None or model_execution_version != "vllm_0_30_0"
                or not fused_embedding or group is None or len(group.traces) != 2
                or hardware_id not in ("nvidia_a800", "nvidia_h200")
                or len(retained_padding_token_ids) != graph_token_shape.padding_tokens):
            raise ValueError("retained padding IDs require a matching fused CUDA TP2 capture and scheduled IDs")
        embedding_ids = input_ids + retained_padding_token_ids
    network = model_type(trace, model, step.pool_name, q_lens, kv_lens, cache_capacity_tokens, slot_mapping,
        group=group, collective=collective, token_ids=embedding_ids, mlp_factories=mlp_factories,
        fused_embedding=fused_embedding, rope_program=rope_program, hardware_id=hardware_id, ffn_factory=ffn_factory,
        model_execution_version=model_execution_version,
        **({"maca_copy_runtime_source": maca_copy_runtime_source} if maca_copy_runtime_source is not None else {}))
    if ascend_piecewise_source:
        # The configured PIECEWISE source also governs overflow NONE calls:
        # using_paged_attention is false unless configured FULL_DECODE_ONLY.
        for layer in network.layers:
            layer.self_attn.attn.graph_attention_source = "ascend_0_18_piecewise_fia"
    # gpu_model_runner allocates INT32 IDs and passes its slice unchanged
    # through _preprocess/_model_forward for text-only models.
    physical_rows = rows if graph_token_shape is None else graph_token_shape.model_input_tokens
    ids = trace.tensor("input_ids", (physical_rows,), "int32")
    pos = trace.tensor("positions", (physical_rows,), "int64")
    if graph_token_shape is not None:
        if model_type not in (Qwen3Model, LlamaModel):
            raise NotImplementedError("source capture requires the ordinary dense Qwen/Llama forward")
        if hardware_id == "ascend_910b" and model_execution_version != "vllm_0_18_0":
            raise NotImplementedError("Ascend captured model requires its explicit0.18 source")
        slots = trace.tensor("capture.slot-mapping", (physical_rows,), "int64")
        for layer in network.layers:
            layer.self_attn.attn.graph_token_shape = graph_token_shape
            layer.self_attn.attn.slot_tensor = slots
            if model_execution_version == "vllm_0_18_0":
                if ascend_piecewise_source:
                    if set(graph_eager_kinds) != {"flash_attn", "kv_update"}:
                        raise ValueError("Ascend PIECEWISE source requires eager attention and KV update")
                elif graph_token_shape.runtime_mode != "FULL" or hardware_id != "ascend_910b":
                    raise NotImplementedError("Ascend0.18 captured attention requires configured FULL")
                else:
                    layer.self_attn.attn.graph_attention_source = "ascend_0_18_full_fia"
        if af_transport is None:
            trace.begin_capture(key=f"{source_profile}:{graph_token_shape.runtime_mode}:{physical_rows}", logical_tokens=rows,
                model_input_tokens=physical_rows, inputs={"input_ids": ids, "positions": pos, "slot_mapping": slots},
                runtime_mode=graph_token_shape.runtime_mode, eager_kinds=graph_eager_kinds)
        else:
            af_transport.attention_capture_inputs[trace.rank] = {"positions": pos, "slot_mapping": slots}
            af_transport.begin_local_capture(trace, {"input_ids": ids}, boundary="entry")
    indices = tuple(sum(q_lens[:i + 1]) - 1 for i in range(len(q_lens)))
    logits_indices = trace.tensor("logits_indices", (len(indices),), "int32")
    hidden_states = network(ids, pos)
    if graph_token_shape is not None:
        if af_transport is None:
            trace.end_capture(hidden_states)
        else:
            af_transport.end_local_capture(trace, {"hidden_states": hidden_states})
    selected, logits, output = _dense_forward_target(trace, model, step.pool_name, hidden_states, logits_indices, indices,
        group=group, gather=gather, model_execution_version=model_execution_version, hardware_id=hardware_id,
        sample_logits=sample_logits, cast_logits=cast_logits, maca_copy_runtime_source=maca_copy_runtime_source)
    return trace


def ascend018_segment_source_scope(trace, *, model, source_profile, model_execution_version,
        tensor_parallel_size, source_lifetime_policy, source_compilation_options):
    """Bind only the declared 0.18 dense TP1 attention-partition source policy.

    The optional options tuple carries upstream compile fields as immutable
    key/value pairs, not a new production configuration/schema. Every field
    is mandatory for this explicit policy; defaults never select a backend.
    """
    if (source_lifetime_policy != "vllm_0_18_npugraph_ex_piecewise"
            or model_execution_version != "vllm_0_18_0"
            or source_profile != "vllm_ascend_0_18_no_metadata_head_only"
            or model.model_type != "qwen3" or model.compute_dtype != "bf16"
            or tensor_parallel_size != 1):
        raise NotImplementedError("segment lifetime policy requires exact Qwen3 BF16 TP1 Ascend0.18 scope")
    required = {"mode": "VLLM_COMPILE", "backend": "vllm_ascend.compilation.compiler_interface.AscendCompiler",
        "cudagraph_mode": "PIECEWISE", "cudagraph_copy_inputs": False,
        "use_inductor_graph_partition": False, "enable_npugraph_ex": True}
    if (not isinstance(source_compilation_options, tuple)
            or any(not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str)
                for pair in source_compilation_options)):
        raise ValueError("source policy requires explicit immutable upstream compilation options")
    options = dict(source_compilation_options)
    if len(options) != len(source_compilation_options) or set(options) != {*required, "splitting_ops"}:
        raise ValueError("source compilation option fields must be complete and unique")
    for name, expected in required.items():
        if type(options[name]) is not type(expected) or options[name] != expected:
            raise ValueError("source compilation option is outside the declared segment policy: " + name)
    splitting = options["splitting_ops"]
    if (not isinstance(splitting, tuple) or not splitting
            or any(not isinstance(name, str) or not name for name in splitting)
            or len(set(splitting)) != len(splitting)
            or "vllm::unified_attention_with_output" not in splitting):
        raise ValueError("source compilation options must explicitly split the attention-with-output op")
    # v0.18 CompilationConfig attention/cache split names. Other members do
    # not occur in this exact Qwen3 source; arbitrary new cuts are rejected.
    allowed_splits = {"vllm::unified_attention", "vllm::unified_attention_with_output",
        "vllm::unified_mla_attention", "vllm::unified_mla_attention_with_output",
        "vllm::mamba_mixer2", "vllm::mamba_mixer", "vllm::short_conv", "vllm::linear_attention",
        "vllm::plamo2_mamba_mixer", "vllm::gdn_attention_core", "vllm::olmo_hybrid_gdn_full_forward",
        "vllm::kda_attention", "vllm::sparse_attn_indexer", "vllm::rocm_aiter_sparse_attn_indexer",
        "vllm::unified_kv_cache_update", "vllm::unified_mla_kv_cache_update", "vllm::mla_forward"}
    if not set(splitting) <= allowed_splits:
        raise ValueError("unrepresented split operators cannot borrow the attention partition")
    splits = tuple(node.node_id for node in trace.graph.nodes if "source_operation" in node.attributes
        and node.attributes["source_operation"] == "flash_attention_profile_fill")
    if not splits:
        raise ValueError("declared source graph contains no bound attention split operations")
    return {"split_node_ids": splits, "escaped_allocation_ids": (),
        "start_event": trace.checkpoints["worker.profile.begin"], "end_event": trace.checkpoints["worker.profile.end"],
        "segment_start_event": trace.checkpoints["model.source.begin"],
        "segment_end_event": trace.checkpoints["model.source.end"]}


def trace_qwen3_startup_profile(model, *, tokens, max_requests, source_profile, num_compute_units=None,
        tensor_parallel_size=1, collective_configs=(), model_execution_version="vllm_0_30_0",
        record_model_scope=False):
    """Version-bound dense startup caller storage for Qwen3 and Llama.

    Conditions select FA's no-metadata output and PyTorch-sort, cold-cache
    native Triton sampling, or the explicit0.11/FlashInfer0.4.1 top-k-first
    storage path. Source versions do not identify a historical library version
    or bind a compiled configuration merely from its backend name.
    Preallocated input buffers and RoPE cache are persistent, not transient
    headroom. Library workspace and other worker state are not inferred.
    The explicit profiles use an unquantized model-dtype head, without a
    head_dtype override. CUDA Sampler.forward casts its logits to FP32.
    Ascend0.18's dummy sampler only gathers hidden rows and calls the head,
    retaining its BF16 logits and never invoking a sampling algorithm.
    FP16 Qwen3 is bound only for the PyTorch-sort branch.
    """
    from .traced_sampling import (TritonSamplingStorageCache, VLLM011WorkerStorage, VLLM024WorkerStorage, VLLM030WorkerStorage,
        worker_profile_storage_forward, ascend018_profile_storage_forward)
    from src.serving.deployment.deployment_config import CollectiveConfig
    if model.compute_dtype != "bf16" and not (
            model.compute_dtype == "fp16" and model.weight_dtype == "fp16"
            and model.model_type == "qwen3"
            and source_profile == "cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise"):
        raise NotImplementedError("FP16 startup sampler storage source requires homogeneous Qwen3 and explicit PyTorch-sort")
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
        raise NotImplementedError("dense startup requires its version-bound0.11,0.18,0.24 or0.30 source")
    if type(tensor_parallel_size) is not int or tensor_parallel_size not in (1, 2):
        raise NotImplementedError("dense startup source requires TP1/TP2")
    if not isinstance(collective_configs, tuple):
        raise TypeError("startup collectives require canonical configuration tuple")
    if tensor_parallel_size == 1:
        if collective_configs:
            raise ValueError("TP1 startup does not consume collective configuration")
        collective = gather = None
    else:
        if (len(collective_configs) != 2 or any(not isinstance(item, CollectiveConfig) for item in collective_configs)
                or tuple(item.operation for item in collective_configs) != ("all_reduce", "all_gather")):
            raise ValueError("TP2 startup requires explicit all-reduce/all-gather configuration")
        collective, gather = collective_configs
    ascend018 = model_execution_version == "vllm_0_18_0"
    metax024 = model_execution_version == "vllm_0_24_0"
    if metax024 and (source_profile != "vllm_metax_0_24_no_metadata_native_triton_fp32_noise_cold_cache"
            or tensor_parallel_size != 1):
        raise NotImplementedError("MetaX0.24 startup requires its explicit BF16 TP1 native-sampler source")
    if ascend018 and source_profile != "vllm_ascend_0_18_no_metadata_head_only":
        raise NotImplementedError("Ascend0.18 startup requires its head-only source profile")
    if source_profile == "vllm_ascend_0_18_no_metadata_head_only":
        if not ascend018 or tensor_parallel_size != 1 or num_compute_units is not None:
            raise NotImplementedError("Ascend head-only startup requires BF16 TP1 and explicit0.18 source")
        sampler_profile = None
    elif source_profile == "vllm_metax_0_24_no_metadata_native_triton_fp32_noise_cold_cache":
        if not metax024 or type(num_compute_units) is not int or num_compute_units <= 0:
            raise ValueError("MetaX0.24 native sampler requires its version and positive compute-unit count")
        sampler_profile = "vllm_0_24_native_triton_or_sort_mixed_no_logprobs_fp32_noise"
    elif source_profile == "cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise":
        if num_compute_units is not None:
            raise ValueError("PyTorch-sort source profile does not consume an SM count")
        sampler_profile = ("vllm_0_11_pytorch_sort_mixed_no_logprobs_fp32_noise"
            if model_execution_version == "vllm_0_11_0" else "vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise")
    elif source_profile == "cuda_eager_inplace_fa_no_metadata_flashinfer_0_4_1_top_k_first":
        if model_execution_version != "vllm_0_11_0":
            raise NotImplementedError("FlashInfer0.4.1 startup storage requires the explicit0.11 worker source")
        if num_compute_units is not None:
            raise ValueError("FlashInfer top-k-first storage does not consume an SM count")
        sampler_profile = "vllm_0_11_flashinfer_0_4_1_top_k_first_mixed_no_logprobs"
    elif source_profile == "cuda_eager_inplace_fa_no_metadata_native_triton_fp32_noise_cold_cache":
        if model_execution_version != "vllm_0_30_0":
            raise NotImplementedError("dense0.11 startup requires the explicit PyTorch-sort sampler source")
        if type(num_compute_units) is not int or num_compute_units <= 0:
            raise ValueError("Triton startup source profile requires an explicit positive SM count")
        sampler_profile = "vllm_0_30_triton_mixed_no_logprobs_fp32_noise"
    else:
        raise NotImplementedError("startup profile requires explicit supported source/backend choices")
    if model.model_type == "qwen3":
        _validate_qwen3_source(model)
        model_type = Qwen3Model
    elif model.model_type == "llama":
        _validate_llama_source(model)
        model_type = LlamaModel
    else:
        raise NotImplementedError("dense startup source requires supported Qwen3 or BF16 Llama")
    if any(type(value) is not int or value <= 0 for value in (tokens, max_requests)):
        raise ValueError("startup profile token and request capacities must be positive integers")
    if tokens > model.max_position_embeddings:
        raise NotImplementedError("initial startup source requires token capacity within the RoPE cache")
    rows = min(tokens, max_requests)
    lengths = (tokens // rows,) * (rows - 1) + (tokens // rows + tokens % rows,)
    traces = tuple(ForwardTrace(rank=rank, namespace=None if tensor_parallel_size == 1 else f"startup:rank-{rank}")
        for rank in range(tensor_parallel_size))
    group = None if tensor_parallel_size == 1 else ForwardCollectiveGroup(traces, namespace="startup")
    # Reuse the existing backend-specific norm/RoPE source modules. These
    # caller scopes do not claim compiler buffer-reuse or library workspace.
    def rank_forward(trace):
        hardware_id = "ascend_910b" if ascend018 else "metax_c500" if metax024 else "nvidia_h200"
        network = model_type(trace, model, "main", lengths, lengths, tokens, None,
            profile_no_metadata=True, rope_program=vllm_rope_program(hardware_id, model.rope_scaling_type),
            hardware_id=hardware_id, group=group, collective=collective,
            model_execution_version=model_execution_version,
            fused_embedding=not (model_execution_version == "vllm_0_11_0" and group is not None),
            # CpuGpuBuffer initializes input_ids to zero. This dense DP1
            # profile has no MoE-driven randomize_inputs branch; the tuple
            # records those source metadata values without numerical work.
            token_ids=None if group is None else (0,) * tokens)
        if ascend018:
            # Only the consumed source ports are needed to measure transient
            # headroom. This does not infer the full NPU worker residency.
            input_ids = trace.tensor("profile.input_ids", (tokens,), "int32", persistent=True)
            positions = trace.tensor("profile.positions", (tokens,), "int64", persistent=True)
        else:
            storage_type = (VLLM011WorkerStorage if model_execution_version == "vllm_0_11_0" else
                VLLM024WorkerStorage if metax024 else VLLM030WorkerStorage)
            worker_storage = storage_type(trace, tokens=tokens, max_requests=max_requests,
                inputs_embeds_size=model.hidden_size, dtype=model.compute_dtype)
            input_ids, positions = worker_storage.input_ids, worker_storage.positions
        if group is None:
            head = SymbolicLinear(trace, "profile.lm-head", "main", model.hidden_size,
                model.vocab_size, kind=WorkloadKind.LM_HEAD)
            compute_logits = lambda x: head(x)[0]
        else:
            # Platform.use_all_gather is true for CUDA; every TP rank's
            # dummy sampler consumes full-vocabulary logits after gathering.
            compute_logits = SymbolicTPLogitsProcessor(trace, "profile.lm-head", "main",
                hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather,
                model_execution_version=model_execution_version)
        cache = None if num_compute_units is None else TritonSamplingStorageCache(trace,
            num_compute_units=num_compute_units, model_execution_version="vllm_0_24_0" if metax024 else "vllm_0_30_0")
        trace.checkpoint("worker.profile.begin")
        if type(record_model_scope) is not bool:
            raise TypeError("model scope recording must be explicit boolean")
        model_forward = network
        if record_model_scope:
            def model_forward(*args, **kwargs):
                trace.checkpoint("model.source.begin")
                result = network(*args, **kwargs)
                trace.checkpoint("model.source.end")
                return result
        if ascend018:
            ascend018_profile_storage_forward(trace, model_forward, compute_logits,
                input_ids=input_ids, positions=positions, query_lengths=lengths)
        else:
            worker_profile_storage_forward(trace, network, compute_logits,
                input_ids=input_ids, positions=positions, query_lengths=lengths,
                sampler_profile=sampler_profile, triton_cache=cache, model_execution_version=model_execution_version)
        trace.checkpoint("worker.profile.end")

    for trace in traces:
        rank_forward(trace)
    return traces[0] if group is None else group


def _ordinary_model_only_source_scope(model, deployment, runtime_inputs):
    """Existing static LM-head scope, shared by binding and graph execution."""
    if (runtime_inputs.model_execution_version != "vllm_0_30_0"
            or runtime_inputs.symbolic_forward_source not in (None, "cuda_eager_inplace")
            or runtime_inputs.target_sampling_source is not None
            or deployment.mode != "colocated" or len(deployment.worker_pools) != 1):
        return False
    pool, = deployment.worker_pools.values()
    parallel = pool.parallel_plan.config
    moe = (model.model_type == "qwen3_moe" and pool.hardware_id == "nvidia_h200"
        and runtime_inputs.expert_routing.source_kind in ("explicit_assumption", "profile_observation"))
    dense = (model.model_type in ("qwen3", "llama") and not model.num_experts
        and runtime_inputs.expert_routing.source_kind == "not_applicable")
    ep2 = (moe and parallel.tensor_parallel_size == parallel.expert_parallel_size == 2
        and parallel.enable_expert_parallel and is_validated_qwen3_moe_ep2_model_scope(model))
    tp2 = (pool.hardware_id == "nvidia_h200" and parallel.tensor_parallel_size == 2
        and (dense or ep2 or moe and not parallel.enable_expert_parallel))
    return ((dense or moe) and model.compute_dtype == model.weight_dtype
        and model.compute_dtype in ("bf16", "fp16")
        and (model.compute_dtype != "fp16" or dense and model.model_type == "qwen3")
        and pool.hardware_id in ("nvidia_a800", "nvidia_h200")
        and len(pool.ranks) == (2 if tp2 else 1) and parallel.is_rank_local
        and (not parallel.enable_expert_parallel or ep2) and parallel.tensor_parallel_size == (2 if tp2 else 1)
        and all(value == 1 for value in (parallel.pipeline_parallel_size,
            parallel.sequence_parallel_size, parallel.context_parallel_size, parallel.data_parallel_size))
        and not runtime_inputs.graph_execution.enabled and runtime_inputs.model_source_inputs is None
        and runtime_inputs.mtp_draft_sampling_source is None and runtime_inputs.mtp_proposer_execution_source is None)


def _ordinary_mla_pd_source_scope(model, deployment):
    """Existing H200 MLA PD pools whose per-pool source is already traced.

    This concerns model execution only. Request handoff still belongs to
    the existing PD/KV owner and does not become a native-provider receipt.
    """
    from src.serving.deployment.deployment_config import worker_role_phase
    if (deployment.mode != "pd_disagg" or model.compute_dtype != "bf16"
            or model.attention_layout not in ("mla", "dsa_mla")
            or not ((model.model_type == "glm_moe_dsa" and model.weight_dtype in ("bf16", "fp8"))
                or (model.model_type == "kimi_k2" and model.model_id == "moonshotai/Kimi-K2.6"
                    and model.weight_dtype == "mixed_int4_bf16"))):
        return False
    pools = tuple(deployment.worker_pools.values())
    if {worker_role_phase(pool.role) for pool in pools} != {"prefill", "decode"}:
        return False
    for pool in pools:
        config = pool.parallel_plan.config
        if (pool.hardware_id != "nvidia_h200" or len(pool.ranks) != 8
                or config.tensor_parallel_size != 8 or not config.is_rank_local
                or (model.model_type == "kimi_k2" and not config.enable_expert_parallel)
                or any(value != 1 for value in (config.pipeline_parallel_size,
                    config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
                or (config.enable_expert_parallel and config.expert_parallel_size != 8)):
            return False
    return True


def _ordinary_af_capture_scope(model, deployment, model_execution_version):
    pools = tuple(deployment.worker_pools.values())
    return (model_execution_version == "vllm_0_11_0"
        and model.model_type in ("qwen3", "llama") and not model.num_experts
        and model.compute_dtype == model.weight_dtype == "bf16"
        and deployment.mode in ("af_disagg", "pd_af_disagg")
        and all(pool.hardware_id == "nvidia_a800" and pool.parallel_plan.config.tensor_parallel_size in (1, 2)
            and pool.parallel_plan.config.pipeline_parallel_size == 1
            and pool.parallel_plan.config.is_rank_local and not pool.parallel_plan.config.enable_expert_parallel
            and len(pool.ranks) == pool.parallel_plan.config.tensor_parallel_size for pool in pools)
        and (deployment.mode != "pd_af_disagg" or all(pool.parallel_plan.config.tensor_parallel_size == 1 for pool in pools)))


def _validate_maca_copy_runtime_scope(model, deployment, runtime_inputs):
    if runtime_inputs.maca_copy_runtime_source is None:
        return
    if (deployment.mode != "colocated" or len(deployment.worker_pools) != 1
            or model.model_type not in ("qwen3", "llama") or model.num_experts):
        raise NotImplementedError("MACA Copy runtime source requires ordinary C5000.24 dense TP1")
    pool, = deployment.worker_pools.values()
    config = pool.parallel_plan.config
    if (pool.hardware_id != "metax_c500" or any(value != 1 for value in (
            config.tensor_parallel_size, config.pipeline_parallel_size,
            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
            or config.enable_expert_parallel):
        raise NotImplementedError("MACA Copy runtime source requires ordinary C5000.24 dense TP1")


def bind_ordinary_forward_source(model, deployment, runtime_inputs):
    """Resolve complete ordinary API facts without retrying a legacy graph."""
    _validate_maca_copy_runtime_scope(model, deployment, runtime_inputs)
    bound = _bind_ordinary_forward_source(model, deployment, runtime_inputs)
    if (bound.symbolic_forward_source is None
            and (bound.symbolic_source_max_model_len is not None
                or bound.symbolic_source_max_batched_tokens is not None)):
        raise NotImplementedError("explicit MLA constructor inputs have no supported common forward; legacy fallback is forbidden")
    if (bound.symbolic_forward_source is None
            and bound.model_execution_version in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0")):
        raise NotImplementedError("declared vLLM source API inputs have no supported common forward; legacy fallback is forbidden")
    return bound


def _bind_ordinary_forward_source(model, deployment, runtime_inputs):
    """Bind an existing ordinary source from complete explicit API settings.

    This selects no sampler, token value, provider or constructor state. An
    already selected source is immutable; unsupported source inputs still
    fail in the shared forward rather than retrying the old builder.
    """
    from dataclasses import replace
    if (runtime_inputs.symbolic_forward_source is None
            and model.model_type in ("glm_moe_dsa", "kimi_k2") and model.compute_dtype == "bf16"
            and runtime_inputs.serving_framework == "vllm"
            and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.attention_kv_layout == "mla" and runtime_inputs.attention_kv_distribution == "uniform"
            and runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            and runtime_inputs.symbolic_source_max_model_len is not None
            and runtime_inputs.symbolic_source_max_batched_tokens is not None
            and (not runtime_inputs.graph_execution.enabled or (deployment.mode == "colocated"
                and len(deployment.worker_pools) == 1 and all(_kimi_single_rank_source_scope(model, pool)
                    for pool in deployment.worker_pools.values())))
            and runtime_inputs.model_source_inputs is None
            and runtime_inputs.mtp_draft_sampling_source is None and runtime_inputs.mtp_proposer_execution_source is None
            and ((deployment.mode == "colocated" and len(deployment.worker_pools) == 1)
                or _ordinary_mla_pd_source_scope(model, deployment))):
        a800_glm_pipeline = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and model.model_type == "glm_moe_dsa" and model.weight_dtype == "bf16"
            and all(pool.hardware_id == "nvidia_a800"
                and pool.parallel_plan.config.tensor_parallel_size == 8
                and pool.parallel_plan.config.pipeline_parallel_size == 4
                and pool.parallel_plan.config.enable_expert_parallel
                and pool.parallel_plan.config.expert_parallel_size == 8
                for pool in deployment.worker_pools.values()))
        a800_kimi_pipeline = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and all(_kimi_pipeline_source_scope(model, pool) for pool in deployment.worker_pools.values()))
        if (a800_glm_pipeline or a800_kimi_pipeline
                or all(_kimi_single_rank_source_scope(model, pool) for pool in deployment.worker_pools.values())
                or all(pool.hardware_id == "nvidia_h200" for pool in deployment.worker_pools.values())):
            if model.model_type == "glm_moe_dsa" and model.weight_dtype in ("bf16", "fp8"):
                profile = ("vllm_0_30_v32_bf16_sync_explicit_allreduce" if model.weight_dtype == "bf16"
                    else "vllm_0_30_v32_fp8_sync_explicit_allreduce")
                return replace(runtime_inputs, symbolic_forward_source=profile)
            if (model.model_type == "kimi_k2" and model.model_id == "moonshotai/Kimi-K2.6"
                    and model.weight_dtype == "mixed_int4_bf16"):
                return replace(runtime_inputs, symbolic_forward_source="vllm_0_30_k2_6_int4_sync_explicit_allreduce")
    if (runtime_inputs.symbolic_forward_source is None and runtime_inputs.serving_framework == "vllm"
            and runtime_inputs.attention_kv_layout == "paged_varlen"
            and runtime_inputs.attention_kv_distribution == "uniform"
            and _ordinary_model_only_source_scope(model, deployment, runtime_inputs)):
        # Missing sampling facts retain the static output boundary. The
        # complete Serving entry still rejects a model-only graph.
        return replace(runtime_inputs, symbolic_forward_source="cuda_eager_inplace")
    if (runtime_inputs.symbolic_forward_source is None and model.model_type == "deepseek_v4"
            and model.compute_dtype == "bf16" and model.weight_dtype == "fp8"
            and runtime_inputs.serving_framework == "vllm"
            and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            and runtime_inputs.expert_routing.source_kind in ("explicit_assumption", "profile_observation")
            and runtime_inputs.attention_kv_layout == "mla" and runtime_inputs.attention_kv_distribution == "uniform"
            and runtime_inputs.v4_eager_metadata_source == "vllm_0_30_eager_dict_all_valid"
            and runtime_inputs.v4_routing_source in ("vllm_0_30_modular_cuda_top6_pure_text", "vllm_0_30_triton_top6_pure_text")
            and runtime_inputs.v4_source_max_model_len is not None
            and runtime_inputs.v4_source_max_batched_tokens is not None
            and runtime_inputs.v4_source_use_deep_gemm is not None
            and runtime_inputs.embedding_token_ids is not None
            and not runtime_inputs.graph_execution.enabled and runtime_inputs.model_source_inputs is None
            and runtime_inputs.mtp_draft_sampling_source is None
            and runtime_inputs.mtp_proposer_execution_source is None
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1):
        pool, = deployment.worker_pools.values()
        parallel = pool.parallel_plan.config
        if (pool.hardware_id == "nvidia_h200" and pool.role == "colocated"
                and parallel.tensor_parallel_size in (4, 8) and parallel.pipeline_parallel_size == 1
                and len(pool.ranks) == parallel.tensor_parallel_size
                and parallel.is_rank_local and not parallel.enable_expert_parallel):
            return replace(runtime_inputs, symbolic_forward_source="vllm_0_30_v4_sync_fp8_ds_mla")
    dense = (model.model_type in ("qwen3", "llama") and not model.num_experts
        and runtime_inputs.expert_routing.source_kind == "not_applicable")
    moe = (model.model_type == "qwen3_moe" and model.num_experts in (2, 128)
        and runtime_inputs.expert_routing.source_kind in ("explicit_assumption", "profile_observation"))
    ordinary_versions = {"vllm_0_11_0": "vllm_0_11", "vllm_0_18_0": "vllm_0_18", "vllm_0_24_0": "vllm_0_24", "vllm_0_30_0": "vllm_0_30"}
    if runtime_inputs.model_execution_version not in ordinary_versions:
        return runtime_inputs
    sampler_prefix = ordinary_versions[runtime_inputs.model_execution_version]
    source011 = runtime_inputs.model_execution_version == "vllm_0_11_0"
    source024 = runtime_inputs.model_execution_version == "vllm_0_24_0"
    source018 = runtime_inputs.model_execution_version == "vllm_0_18_0"
    ordinary_moe_capture = ((source011 or source024 or source018) and moe and deployment.mode == "colocated"
        and len(deployment.worker_pools) == 1
        and all(_versioned_qwen3_moe_ep2_source(model, item, runtime_inputs.model_execution_version) for item in deployment.worker_pools.values()))
    if (runtime_inputs.symbolic_forward_source is not None
            or runtime_inputs.serving_framework != "vllm"
            or runtime_inputs.target_sampling_source != sampler_prefix + "_plain_greedy_" + model.compute_dtype
            or not (dense or moe) or (model.compute_dtype == "fp16" and model.model_type != "qwen3")
            or model.compute_dtype not in ("bf16", "fp16") or model.weight_dtype != model.compute_dtype
            or runtime_inputs.attention_kv_layout != "paged_varlen"
            or runtime_inputs.attention_kv_distribution != "uniform"
            or (runtime_inputs.graph_execution.enabled and not (_ordinary_af_capture_scope(model, deployment, runtime_inputs.model_execution_version)
                or (source011 or source024 or source018) and (dense or ordinary_moe_capture)
                and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
                and all(pool.hardware_id in (("metax_c500",) if source024 else ("ascend_910b",) if source018 else ("nvidia_a800", "nvidia_h200"))
                    for pool in deployment.worker_pools.values())))
            or runtime_inputs.model_source_inputs is not None
            or runtime_inputs.mtp_draft_sampling_source is not None
            or runtime_inputs.mtp_proposer_execution_source is not None
            or runtime_inputs.v4_eager_metadata_source is not None):
        return runtime_inputs
    if (source024 or source018) and (not (dense or ordinary_moe_capture) or deployment.mode != "colocated"
            or len(deployment.worker_pools) != 1
            or any(pool.hardware_id != ("metax_c500" if source024 else "ascend_910b") for pool in deployment.worker_pools.values())):
        return runtime_inputs
    if source011 and moe and deployment.mode in ("pd_disagg", "af_disagg"):
        # These existing CLI modes already execute this version-bound MoE
        # forward. Bind the same complete facts for programmatic callers;
        # the forward still validates roles, routes and per-step inputs.
        pools = tuple(deployment.worker_pools.values())
        if (model.compute_dtype == "bf16" and len(pools) == 2
                and all(item.hardware_id == "nvidia_a800" and len(item.ranks) == 1
                    and item.parallel_plan.config.tensor_parallel_size == 1
                    and item.parallel_plan.config.pipeline_parallel_size == 1
                    and item.parallel_plan.config.is_rank_local
                    and not item.parallel_plan.config.enable_expert_parallel for item in pools)):
            return replace(runtime_inputs, symbolic_forward_source="cuda_eager_inplace")
        return runtime_inputs
    if dense and deployment.mode in ("pd_disagg", "af_disagg", "pd_af_disagg"):
        # The already adapted per-pool model consumes the same declared
        # sampler in these existing A800 modes. IDs still belong to each
        # step; the forward validates transfers, ownership and pool roles.
        pools = tuple(deployment.worker_pools.values())
        configs = tuple(item.parallel_plan.config for item in pools)
        if (model.compute_dtype == "bf16"
                and all(item.hardware_id == "nvidia_a800" for item in pools)
                and len({config.tensor_parallel_size for config in configs}) == 1
                and all(config.tensor_parallel_size in (1, 2) and config.pipeline_parallel_size == 1
                    and config.is_rank_local and not config.enable_expert_parallel for config in configs)):
            return replace(runtime_inputs, symbolic_forward_source="cuda_eager_inplace")
        return runtime_inputs
    if dense and model.compute_dtype == "bf16" and deployment.mode == "colocated" and len(deployment.worker_pools) > 1:
        pools = tuple(deployment.worker_pools.values())
        configs = tuple(item.parallel_plan.config for item in pools)
        if (all(item.hardware_id == "nvidia_a800" and item.role == "colocated" for item in pools)
                and len({config.tensor_parallel_size for config in configs}) == 1
                and all(config.tensor_parallel_size in (1, 2) and config.pipeline_parallel_size == 1
                    and config.is_rank_local and not config.enable_expert_parallel for config in configs)):
            return replace(runtime_inputs, symbolic_forward_source="cuda_eager_inplace")
        return runtime_inputs
    if deployment.mode != "colocated" or len(deployment.worker_pools) != 1:
        return runtime_inputs
    pool, = deployment.worker_pools.values()
    parallel = pool.parallel_plan.config
    pipeline = (dense and pool.hardware_id == "nvidia_a800" and not parallel.enable_expert_parallel
        and is_validated_qwen3_pipeline_model_scope(model, parallel.tensor_parallel_size, parallel.pipeline_parallel_size))
    hardware_ids = (("nvidia_a800", "nvidia_h200") if source011
        else ("metax_c500",) if source024 else ("ascend_910b",) if source018
        else ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b"))
    if (pool.hardware_id not in hardware_ids or pool.role != "colocated"
            or (source011 and pool.hardware_id == "nvidia_h200" and model.compute_dtype != "bf16")
            or ((source024 or source018) and (not (dense or ordinary_moe_capture) or model.compute_dtype != "bf16"))
            or parallel.tensor_parallel_size not in (1, 2) or (parallel.pipeline_parallel_size != 1 and not pipeline)
            or not parallel.is_rank_local or dense and parallel.enable_expert_parallel):
        return runtime_inputs
    if moe and (pool.hardware_id not in (("nvidia_a800", "nvidia_h200") if source011 else ("metax_c500",) if source024 else ("ascend_910b",) if source018 else ("nvidia_h200",))
            or (source011 and pool.hardware_id == "nvidia_h200" and not _vllm011_qwen3_moe_ep2_source(model, pool)) or not (
            (parallel.tensor_parallel_size == 1 or not source011 and parallel.tensor_parallel_size == 2)
                and not parallel.enable_expert_parallel
            or parallel.tensor_parallel_size == parallel.expert_parallel_size == 2
                and parallel.enable_expert_parallel and is_validated_qwen3_moe_ep2_model_scope(model))):
        return runtime_inputs
    if moe and model.compute_dtype != "bf16":
        return runtime_inputs
    return replace(runtime_inputs, symbolic_forward_source=(
        "vllm_ascend_eager" if pool.hardware_id == "ascend_910b" else "cuda_eager_inplace"))


def _dense_source_prefix(model, step, arguments):
    """Execute the real first pipeline range of the same dense model source."""
    q_lens, kv_lens, rows = _dense_source_metadata(model, step,
        input_ids=arguments["input_ids"], positions=arguments["positions"],
        slot_mapping=arguments["slot_mapping"], cache_capacity_tokens=arguments["cache_capacity_tokens"],
        source_profile=arguments["source_profile"], hardware_id=arguments["hardware_id"],
        allow_unknown_input_ids=True)
    trace = ForwardTrace(rank=arguments["rank"], namespace=arguments["namespace"])
    network = arguments["model_type"](trace, model, step.pool_name, q_lens, kv_lens,
        arguments["cache_capacity_tokens"], arguments["slot_mapping"],
        token_ids=arguments["input_ids"], layer_range=(0, 1),
        rope_program=arguments["rope_program"], hardware_id=arguments["hardware_id"],
        model_execution_version=arguments["model_execution_version"],
        maca_copy_runtime_source=arguments["maca_copy_runtime_source"])
    shape = arguments["graph_token_shape"]
    ids = trace.tensor("input_ids", (shape.model_input_tokens,), "int32")
    positions = trace.tensor("positions", (shape.model_input_tokens,), "int64")
    slots = trace.tensor("capture.slot-mapping", (shape.model_input_tokens,), "int64")
    for layer in network.layers:
        layer.self_attn.attn.graph_token_shape = shape
        layer.self_attn.attn.slot_tensor = slots
        if arguments["model_execution_version"] == "vllm_0_18_0":
            layer.self_attn.attn.graph_attention_source = "ascend_0_18_full_fia"
    trace.begin_capture(key=f"{arguments['source_profile']}:FULL:{shape.model_input_tokens}",
        logical_tokens=rows, model_input_tokens=shape.model_input_tokens,
        inputs={"input_ids": ids, "positions": positions, "slot_mapping": slots})
    output = network(ids, positions)
    trace.end_capture(output["hidden_states"])
    return trace


def _dense_source_program(model, step, arguments, *, cache, launch_policy):
    """Reuse only the proven ordinary captured dense source program domain."""
    shape = arguments["graph_token_shape"]
    supported = (model.model_type in ("qwen3", "llama") and not model.num_experts
        and model.num_hidden_layers > 1 and model.compute_dtype == model.weight_dtype == "bf16"
        and (arguments["hardware_id"], arguments["model_execution_version"]) in (
            ("metax_c500", "vllm_0_24_0"), ("nvidia_a800", "vllm_0_11_0"),
            ("nvidia_h200", "vllm_0_11_0"),
            ("ascend_910b", "vllm_0_18_0"))
        and shape.runtime_mode == "FULL" and not arguments["graph_eager_kinds"]
        and all(request.phase == "decode" and request.scheduled_tokens == 1
            and request.mtp_proposer_mode == "disabled" for request in step.requests))
    if not supported:
        return _trace_dense_forward(model, step, **arguments)
    return _record_dense_source_program(model, step, arguments, cache=cache, launch_policy=launch_policy)


class _DenseMetadataSourceGuard:
    """Run-local source identity contract, recorded once per source program."""

    def __init__(self):
        import sys
        from types import FunctionType
        from . import graph_padding
        modules = tuple(sys.modules[name] for name in (
            __name__, SymbolicAttention.__module__, ForwardTrace.__module__,
            graph_padding.__name__))
        self.modules = tuple((module, dict(vars(module))) for module in modules)
        classes = {value for module in modules for value in vars(module).values() if isinstance(value, type)}
        self.classes = tuple((cls, self._class_contract(cls)) for cls in classes)
        functions = {value for module in modules for value in vars(module).values()
            if isinstance(value, FunctionType)}
        for cls in classes:
            for member in vars(cls).values():
                if isinstance(member, (classmethod, staticmethod)):
                    member = member.__func__
                if isinstance(member, FunctionType):
                    functions.add(member)
        self.functions = tuple((function, function.__code__, function.__defaults__,
            None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)) for function in functions)

    @staticmethod
    def _class_contract(cls):
        values = dict(vars(cls))
        if cls in (ModelConfig, ForwardTrace, ForwardCollectiveGroup, MemoryAllocation) and "__slotnames__" in values:
            # copy.copy and pickle memoize CPython's empty slots lookup.
            # No source semantics change; nonempty or non-list values reject.
            if type(values["__slotnames__"]) is not list or values["__slotnames__"]:
                return None
            del values["__slotnames__"]
        return values

    def seal(self):
        # Include this very contract in the model-owner namespace snapshot.
        self.modules = tuple((module, dict(vars(module))) for module, _ in self.modules)

    def unchanged(self):
        return (all(vars(module) == snapshot for module, snapshot in self.modules)
            and all(self._class_contract(cls) == snapshot for cls, snapshot in self.classes)
            and all(function.__code__ is code and function.__defaults__ == defaults
                and function.__kwdefaults__ == kwdefaults
                for function, code, defaults, kwdefaults in self.functions))


def _dense_parameter_metadata(model, step, arguments):
    q_lens, kv_lens, _ = _dense_source_metadata(model, step,
        input_ids=arguments["input_ids"], positions=arguments["positions"],
        slot_mapping=arguments["slot_mapping"], cache_capacity_tokens=arguments["cache_capacity_tokens"],
        source_profile=arguments["source_profile"], hardware_id=arguments["hardware_id"],
        allow_unknown_input_ids=True)
    if arguments["model_execution_version"] == "vllm_0_18_0":
        from .graph_padding import GraphAttentionShape
        attention_shape = GraphAttentionShape.from_ascend_full(
            arguments["graph_token_shape"], q_lens, kv_lens, prefill_no_cache=q_lens == kv_lens)
        metadata = {"attention_shape": attention_shape,
            "kv_len": max(attention_shape.model_input_kv_lens),
            "kv_lens": kv_lens, "slot_mapping": arguments["slot_mapping"]}
    else:
        metadata = gqa_full_source_metadata(q_lens, kv_lens, arguments["slot_mapping"], arguments["graph_token_shape"])
    return metadata


def _dense_metadata_prefix(model, step, arguments, program, *, tp_size=1):
    """Bind the recorded actual prefix from the same pure primitive metadata.

    No model operations are synthesized: all nodes, ports, aliases, events and
    lifetimes are retained from its actual source recording. Only the declared
    attention values and persistent KV layout change.
    """
    from copy import copy
    from math import prod
    metadata = _dense_parameter_metadata(model, step, arguments)
    result = copy(program.prefix)
    result._nodes = dict(result._nodes)
    result._allocations, result._views = dict(result._allocations), dict(result._views)
    attention = result.qualify("layer-0:attention")
    node = result._nodes[attention]
    result._nodes[attention] = replace(node, shape={**node.shape, "kv_len": metadata["kv_len"]},
        attributes={**node.attributes, "kv_lens": metadata["kv_lens"],
            "model_input_kv_lens": metadata["attention_shape"].model_input_kv_lens})
    update = attention + ":kv-update"
    result._nodes[update] = result._nodes[update].with_attributes({
        **result._nodes[update].attributes, "symbolic_slot_mapping": metadata["slot_mapping"]})
    cache = result.qualify("layer-0:kv-cache")
    shape = gqa_cache_shape(arguments["cache_capacity_tokens"], model.kv_heads_per_tensor_parallel_rank(tp_size), model.head_dim)
    result._allocations[cache] = replace(result._allocations[cache], size_bytes=prod(shape) * result._views[cache].element_bytes)
    result._views[cache] = ForwardTrace._dense_view(cache, shape, result._views[cache].dtype)
    return result


def _record_dense_source_program(model, step, arguments, *, cache, launch_policy):
    """Record/bind after the caller has established the source program domain."""
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens", "namespace"}
    # All actual static source arguments, including known/unknown token IDs,
    # remain in this run-local key. ModelConfig is an immutable full contract.
    key = (model, step.pool_name, tuple(request.scheduled_tokens for request in step.requests),
        launch_policy, tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    program = cache.lookup(key)
    arguments = dict(arguments)
    arguments["namespace"] = (cache.namespace(model.model_type, step.pool_name,
        arguments["rank"], arguments["model_execution_version"]) if program is None else program.source.namespace)
    metadata_domain = (arguments["model_type"] in (Qwen3Model, LlamaModel)
        and (arguments["hardware_id"], arguments["model_execution_version"]) in (
            ("nvidia_a800", "vllm_0_11_0"), ("nvidia_h200", "vllm_0_11_0"),
            ("metax_c500", "vllm_0_24_0"))
        and arguments["graph_token_shape"].runtime_mode == "FULL")
    metadata_scope = metadata_domain and _DENSE_METADATA_SOURCE_CONTRACT.unchanged()
    seal_storage = (metadata_scope and arguments["hardware_id"] in ("nvidia_a800", "nvidia_h200")
        and model.compute_dtype == model.weight_dtype == "bf16"
        and arguments["source_profile"] == "cuda_eager_inplace")
    if program is not None and metadata_domain:
        if (not metadata_scope or "_dense_metadata_guard" not in program.__dict__
                or program._dense_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
            program = None
    prefix = (_dense_metadata_prefix(model, step, arguments, program)
        if program is not None and metadata_scope else _dense_source_prefix(model, step, arguments))
    if program is not None:
        bound = program.bind(prefix)
        if bound is not None:
            if seal_storage:
                from .traced_forward import _seal_native_bound_storage
                _seal_native_bound_storage(program, bound)
            return bound
    source = _trace_dense_forward(model, step, **arguments)
    if launch_policy == "injected_launch":
        source.bind_launch_policy(launch_policy)
    for name, node in prefix._nodes.items():
        if name not in source._nodes:
            return source
        complete = source._nodes[name]
        if ((node.kind, node.pool_name, node.shape, node.dtype, node.variant, node.rank) !=
                (complete.kind, complete.pool_name, complete.shape, complete.dtype, complete.variant, complete.rank)
                or {key: value for key, value in node.attributes.items() if key != "launch_policy"} !=
                {key: value for key, value in complete.attributes.items() if key != "launch_policy"}):
            # A changed/opaque prefix is still executed by the complete real
            # source. It cannot establish a reusable parameter-binding program.
            return source
    node_bindings, allocation_bindings = [], []
    # The model constructor uses the same homogeneous attention class for
    # every layer. All dynamic values below come from that class's real prefix
    # invocation; static nodes and the complete source DAG are never inferred.
    for layer in range(model.num_hidden_layers):
        target, origin = source.qualify(f"layer-{layer}"), prefix.qualify("layer-0")
        node_bindings.extend(((target + ":attention", origin + ":attention", ("kv_len",),
            ("kv_lens", "model_input_kv_lens")),
            (target + ":attention:kv-update", origin + ":attention:kv-update", (), ("symbolic_slot_mapping",))))
        allocation_bindings.append((target + ":kv-cache", origin + ":kv-cache"))
    program = ForwardSourceProgram(source, prefix,
        node_bindings=node_bindings, allocation_bindings=allocation_bindings)
    if metadata_scope:
        program._dense_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    # Return an independently bound source even on the first invocation;
    # downstream launch binding must never mutate the stored template.
    bound = program.bind(prefix)
    if bound is None:
        raise RuntimeError("newly recorded dense source program rejected its own actual prefix")
    if seal_storage:
        from .traced_forward import _seal_native_bound_storage
        _seal_native_bound_storage(program, bound)
    return bound


def build_symbolic_workload(step, model, deployment, *, launch_policy, runtime_inputs, kv_cache_events=(), hardware_by_id=None,
                            trace_observer=None, source_program_cache=None):
    """Production dispatch from normalized schedule to the shared symbolic graph.

    Slot coordinates describe disjoint request ranges in a declared logical
    cache arena. Physical page allocation remains the Scheduler/KV manager's
    separate contract; this source projection never invents native page IDs.
    """
    from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent, host_kv_transfer_attributes
    from .model_config import validate_named_target_identity
    validate_named_target_identity(model)
    _validate_maca_copy_runtime_scope(model, deployment, runtime_inputs)
    if step.attention_kv_layout is not None and step.attention_kv_layout != runtime_inputs.attention_kv_layout:
        raise ValueError("scheduler attention KV layout disagrees with RuntimeInputs")
    kv_cache_events = tuple(kv_cache_events)
    if trace_observer is not None and not callable(trace_observer):
        raise TypeError("symbolic trace observer must be callable")
    if source_program_cache is not None and not isinstance(source_program_cache, SourceProgramCache):
        raise TypeError("symbolic source programs require their run-local cache owner")

    def finish(source):
        if launch_policy == "injected_launch":
            traces = (source,) if isinstance(source, ForwardTrace) else source.traces
            for trace in traces:
                trace.bind_launch_policy(launch_policy)
        if host_transfers:
            source = ForwardKVCacheGroup(source, pool=pool, step_id=step.step_id, transfer_attributes=host_transfers)
        graph = source.graph
        if trace_observer is not None:
            trace_observer(source, graph)
        return graph
    for event in kv_cache_events:
        if not isinstance(event, KVCacheEvent):
            raise TypeError("symbolic workload requires normalized cache events")
    pool = deployment.worker_pools[step.pool_name]
    parallel = pool.parallel_plan.config
    versioned_moe_source = (runtime_inputs.model_execution_version in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0")
        and runtime_inputs.symbolic_forward_source == ("vllm_ascend_eager" if runtime_inputs.model_execution_version == "vllm_0_18_0" else "cuda_eager_inplace") and model.model_type == "qwen3_moe"
        and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
        and _versioned_qwen3_moe_ep2_source(model, pool, runtime_inputs.model_execution_version))
    moe_capture = runtime_inputs.graph_execution.enabled and versioned_moe_source
    if (model.model_type in ("qwen3", "llama") and runtime_inputs.symbolic_forward_source in ("cuda_eager_inplace", "vllm_ascend_eager")
            and runtime_inputs.target_sampling_source is not None):
        expected_sampler = ("vllm_0_11" if runtime_inputs.model_execution_version == "vllm_0_11_0"
            else "vllm_0_18" if runtime_inputs.model_execution_version == "vllm_0_18_0"
            else "vllm_0_24" if runtime_inputs.model_execution_version == "vllm_0_24_0"
            else "vllm_0_30") + "_plain_greedy_" + model.compute_dtype
        cast_only = (runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16"
            and runtime_inputs.model_execution_version == "vllm_0_30_0" and model.compute_dtype == "bf16")
        if runtime_inputs.target_sampling_source != expected_sampler and not cast_only:
            raise ValueError("ordinary dense sampler source must match the model compute dtype")
    from .graph_padding import VLLMDenseGraphInputs
    dense_graph_inputs = (runtime_inputs.model_source_inputs
        if isinstance(runtime_inputs.model_source_inputs, VLLMDenseGraphInputs) else None)
    if dense_graph_inputs is not None and (
            model.model_type not in ("qwen3", "llama") or model.num_experts
            or parallel.tensor_parallel_size != 2 or parallel.pipeline_parallel_size != 1
            or pool.hardware_id not in ("nvidia_a800", "nvidia_h200")):
        raise NotImplementedError("dense retained input state requires NVIDIA dense TP2/PP1 capture")
    if (runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and runtime_inputs.model_execution_version == "vllm_0_30_0" and runtime_inputs.target_sampling_source is None):
        if (not _ordinary_model_only_source_scope(model, deployment, runtime_inputs)
                or kv_cache_events or any(request.mtp_proposer_mode != "disabled" or any((request.mtp_proposed_tokens,
                    request.mtp_verified_tokens, request.mtp_proposer_tokens, request.mtp_next_proposed_tokens)) for request in step.requests)):
            raise NotImplementedError("static LM-head-only source requires0.30 CUDA dense TP1/H200 dense TP2 or H200 Qwen3-MoE TP1/validated TP=EP2 eager without MTP/graph")
    if runtime_inputs.model_execution_version == "vllm_0_11_0" and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace":
        moe_ep2 = _vllm011_qwen3_moe_ep2_source(model, pool)
        from src.serving.deployment.validated_scope import is_validated_standalone_af_ep2_scope
        moe_disagg = (model.model_type == "qwen3_moe" and not runtime_inputs.graph_execution.enabled
            and ((deployment.mode == "af_disagg" and ((parallel.tensor_parallel_size == 1 and not parallel.enable_expert_parallel)
                or is_validated_standalone_af_ep2_scope(deployment, model)))
                or (deployment.mode == "pd_disagg" and (moe_ep2 or (parallel.tensor_parallel_size == 1
                    and not parallel.enable_expert_parallel)))))
        dense_pipeline = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and is_validated_qwen3_pipeline_model_scope(model, parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
            and not parallel.enable_expert_parallel)
        dense_disagg = (deployment.mode in ("af_disagg", "pd_disagg", "pd_af_disagg") and model.model_type in ("qwen3", "llama")
            and not model.num_experts and (not runtime_inputs.graph_execution.enabled
                or _ordinary_af_capture_scope(model, deployment, runtime_inputs.model_execution_version))
            and not parallel.enable_expert_parallel and parallel.tensor_parallel_size in (1, 2))
        h200_dense = (pool.hardware_id == "nvidia_h200" and model.model_type in ("qwen3", "llama")
            and not model.num_experts and model.compute_dtype == model.weight_dtype == "bf16"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and parallel.pipeline_parallel_size == 1 and parallel.tensor_parallel_size in (1, 2)
            and not parallel.enable_expert_parallel)
        h200_moe = (pool.hardware_id == "nvidia_h200" and model.model_type == "qwen3_moe"
            and moe_ep2 and deployment.mode == "colocated" and len(deployment.worker_pools) == 1)
        if ((pool.hardware_id != "nvidia_a800" and not h200_dense and not h200_moe) or model.model_type not in ("qwen3", "llama", "qwen3_moe")
                or (deployment.mode != "colocated" and not (dense_disagg or moe_disagg)) or (parallel.enable_expert_parallel and not moe_ep2)
                or (parallel.pipeline_parallel_size > 1 and not dense_pipeline)
                or (model.model_type == "qwen3_moe" and ((runtime_inputs.graph_execution.enabled and not moe_capture)
                    or (not moe_ep2 and not moe_disagg and (parallel.tensor_parallel_size != 1 or len(pool.ranks) != 1))))):
            raise NotImplementedError("vLLM0.11 source requires H200 BF16 dense colocated TP1/TP2 PP1 or existing A800 dense/MoE deployment scope")
    if runtime_inputs.model_execution_version in ("vllm_0_18_0", "vllm_0_24_0"):
        if (pool.hardware_id != ("ascend_910b" if runtime_inputs.model_execution_version == "vllm_0_18_0" else "metax_c500") or (model.model_type not in ("qwen3", "llama") and not versioned_moe_source)
                or (model.num_experts and not versioned_moe_source) or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
                or deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                or parallel.tensor_parallel_size not in (1, 2) or parallel.pipeline_parallel_size != 1
                or not parallel.is_rank_local or (parallel.enable_expert_parallel and not versioned_moe_source)):
            raise NotImplementedError("versioned device model source requires matching C5000.24 or Ascend0.18 BF16 dense colocated TP1/TP2 PP1 EP-off")
    intents = tuple(event for event in kv_cache_events if event.action in (KVCacheAction.LOAD, KVCacheAction.EVICT))
    host_transfers = ()
    if intents:
        from src.serving.deployment.validated_scope import is_validated_colocated_tp2_host_kv_scope
        glm4_af_host = (model.model_type == "glm4_moe" and deployment.mode == "af_disagg"
            and pool.hardware_id == "nvidia_a800" and len(pool.ranks) == 1
            and runtime_inputs.symbolic_forward_source == "vllm_0_11_glm4_sync")
        if (model.attention_layout != "standard_gqa"
                or (model.model_type not in ("qwen3", "qwen3_moe", "llama") and not glm4_af_host)
                or parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel or not parallel.is_rank_local):
            raise NotImplementedError("virtual Host KV requires its existing ordinary GQA PP1/EP-off domain")
        if len(pool.ranks) != 1 and not is_validated_colocated_tp2_host_kv_scope(deployment, model):
            raise NotImplementedError("virtual Host KV requires one physical owner or the validated Qwen3 TP2 scope")
        host_transfers = tuple(host_kv_transfer_attributes(event, cache_config=pool.kv_cache,
            model_bytes_per_token=model.kv_bytes_per_token_per_parallel_rank(parallel.tensor_parallel_size,
                parallel.pipeline_parallel_size), scheduled_request_ids={request.request_id for request in step.requests})
            for event in intents)
    if launch_policy == "injected_launch":
        injected_pipeline = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and is_validated_qwen3_pipeline_model_scope(model, parallel.tensor_parallel_size, parallel.pipeline_parallel_size))
        if (runtime_inputs.symbolic_forward_source != ("vllm_ascend_eager" if runtime_inputs.model_execution_version == "vllm_0_18_0" else "cuda_eager_inplace")
                or model.model_type not in ("qwen3", "llama") or model.num_experts
                or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
                or (pool.hardware_id != "nvidia_a800" and not (((pool.hardware_id == "nvidia_h200"
                    and runtime_inputs.model_execution_version == "vllm_0_11_0")
                    or (pool.hardware_id == "metax_c500" and runtime_inputs.model_execution_version == "vllm_0_24_0")
                    or (pool.hardware_id == "ascend_910b" and runtime_inputs.model_execution_version == "vllm_0_18_0"))
                    and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
                    and parallel.pipeline_parallel_size == 1))
                or deployment.mode not in ("colocated", "af_disagg", "pd_disagg", "pd_af_disagg")
                or parallel.tensor_parallel_size not in (1, 2) or (parallel.pipeline_parallel_size != 1 and not injected_pipeline)
                or parallel.enable_expert_parallel or intents) and not versioned_moe_source:
            raise NotImplementedError("virtual injected selection requires ordinary A800 BF16 dense deployment or H200 vLLM0.11 dense colocated TP1/TP2 PP1")
    elif launch_policy != "analytical_launch":
        raise ValueError("virtual source requires an explicit analytical or injected launch policy")
    if runtime_inputs.embedding_token_ids is None:
        # TP1 owns the whole vocabulary; the0.11 TP2 masked program instead
        # gathers all rows before masking/AR. Their public logical payloads
        # depend only on shape. Keep IDs unknown, never dummy or sequential;
        # the fused TP branch still needs actual IDs for shard ownership.
        shape_only_embedding = ((parallel.tensor_parallel_size == 1 and len(pool.ranks) == 1)
            or (parallel.tensor_parallel_size == 2 and len(pool.ranks) == 2
                and ((runtime_inputs.model_execution_version == "vllm_0_11_0" and pool.hardware_id in ("nvidia_a800", "nvidia_h200"))
                    or (runtime_inputs.model_execution_version in ("vllm_0_24_0", "vllm_0_30_0") and pool.hardware_id == "metax_c500")
                    or (runtime_inputs.model_execution_version == "vllm_0_18_0" and pool.hardware_id == "ascend_910b"))))
        shape_only_moe = (model.model_type == "qwen3_moe" and parallel.tensor_parallel_size == 1
            and pool.hardware_id == "nvidia_h200" and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.expert_routing.source_kind in ("explicit_assumption", "profile_observation"))
        shape_only_replicas = (runtime_inputs.model_execution_version == "vllm_0_11_0"
            and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and all(item.role == "colocated" and item.hardware_id == "nvidia_a800"
                and item.parallel_plan.config.tensor_parallel_size in (1, 2)
                and item.parallel_plan.config.pipeline_parallel_size == 1 and item.parallel_plan.config.is_rank_local
                and not item.parallel_plan.config.enable_expert_parallel
                for item in deployment.worker_pools.values()))
        shape_only_dense = (((model.model_type in ("qwen3", "llama") and not model.num_experts
                and runtime_inputs.expert_routing.source_kind == "not_applicable") or shape_only_moe or versioned_moe_source)
            and model.compute_dtype == model.weight_dtype and model.compute_dtype in ("bf16", "fp16")
            and deployment.mode == "colocated" and (len(deployment.worker_pools) == 1 or shape_only_replicas) and shape_only_embedding
            and parallel.pipeline_parallel_size == 1
            and parallel.is_rank_local and (not parallel.enable_expert_parallel or versioned_moe_source)
            and runtime_inputs.serving_framework == "vllm"
            and runtime_inputs.model_execution_version in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0")
            and runtime_inputs.symbolic_forward_source in ("cuda_eager_inplace", "vllm_ascend_eager")
            and runtime_inputs.mtp_draft_sampling_source is None and runtime_inputs.mtp_proposer_execution_source is None
            and runtime_inputs.model_source_inputs is None
            and (not runtime_inputs.graph_execution.enabled or (parallel.tensor_parallel_size in (1, 2)
                and ((runtime_inputs.model_execution_version == "vllm_0_11_0" and pool.hardware_id in ("nvidia_a800", "nvidia_h200"))
                    or (runtime_inputs.model_execution_version == "vllm_0_24_0" and pool.hardware_id == "metax_c500")
                    or (runtime_inputs.model_execution_version == "vllm_0_18_0" and pool.hardware_id == "ascend_910b"))))
            # The Scheduler already resolved prefix hits/misses and supplies
            # the resulting token/KV lengths. Performance tracing needs those
            # shapes, not embedding values or a second token-level cache test.
            # Host transfers retain their separate payload contract.
            and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE,
                KVCacheAction.HIT, KVCacheAction.MISS) for event in kv_cache_events)
            and all(request.mtp_proposer_mode == "disabled" and not any((request.mtp_proposed_tokens,
                request.mtp_verified_tokens, request.mtp_proposer_tokens, request.mtp_next_proposed_tokens))
                for request in step.requests))
        if not shape_only_dense:
            raise ValueError("symbolic forward requires explicit scheduled token IDs outside ordinary shape-only source; no value fallback")
    if runtime_inputs.graph_execution.enabled:
        config = pool.parallel_plan.config
        pipeline_capture = (runtime_inputs.model_execution_version == "vllm_0_11_0"
            and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and pool.hardware_id == "nvidia_a800" and not config.enable_expert_parallel
            and is_validated_qwen3_pipeline_model_scope(model, config.tensor_parallel_size, config.pipeline_parallel_size))
        sglang_capture = (runtime_inputs.serving_framework == "sglang" and model.model_type == "deepseek_v3"
            and pool.hardware_id == "nvidia_h200" and config.tensor_parallel_size == 8
            and len(pool.ranks) == 8 and config.is_rank_local and config.pipeline_parallel_size == 1
            and not config.enable_expert_parallel and deployment.mode == "colocated" and not kv_cache_events)
        host_capture = (runtime_inputs.model_execution_version == "vllm_0_11_0"
            and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace" and pool.hardware_id == "nvidia_a800"
            and model.model_type in ("qwen3", "llama") and not model.num_experts)
        af_capture = (_ordinary_af_capture_scope(model, deployment, runtime_inputs.model_execution_version)
            and (not intents or host_capture))
        kimi_capture = (runtime_inputs.symbolic_forward_source == "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and _kimi_single_rank_source_scope(model, pool)
            and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE) for event in kv_cache_events))
        if not kimi_capture and not af_capture and not sglang_capture and not moe_capture and not pipeline_capture and (model.model_type not in ("qwen3", "llama") or model.num_experts or deployment.mode != "colocated"
                or pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
                or config.tensor_parallel_size not in (1, 2) or len(pool.ranks) != config.tensor_parallel_size
                or any(value != 1 for value in (config.pipeline_parallel_size,
                    config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
                or config.enable_expert_parallel or (intents and not host_capture)):
            raise NotImplementedError("source capture requires NVIDIA dense Qwen/Llama TP1/TP2 PP1 EP-off without Host KV")
    positions, slots, offset = [], [], 0
    for request in step.requests:
        context, query = request.computed_tokens, request.scheduled_tokens
        positions.extend(range(context, context + query))
        slots.extend(range(offset + context, offset + context + query))
        offset += context + query
    if runtime_inputs.symbolic_forward_source in ("sglang_0_5_19_deepseek_v3_decode_fp8",
            "sglang_0_5_19_deepseek_v3_decode_fp8_flashinfer_ar_norm",
            "sglang_0_5_19_deepseek_v3_target_fp8", "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm"):
        from .traced_sglang import (SOURCE_FUSED_PROFILE, SOURCE_TARGET_PROFILE, SOURCE_TARGET_FUSED_PROFILE, PREFILL_PREFIX_SOURCE,
            SGLangPrefillInputs, SGLangGraphInputs, SGLangEagleChainState,
            trace_sglang_deepseek_v3_decode, trace_sglang_deepseek_v3_forward)
        if deployment.mode != "colocated":
            raise NotImplementedError("SGLang source requires colocated deployment")
        # These named profiles declare the Cutlass/FlashMLA source branches
        # and whether FlashInfer AR/residual/RMSNorm fusion is enabled.
        # They do not infer installed providers or runtime flags.
        arguments = dict(input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions),
            slot_mapping=tuple(slots), cache_capacity_tokens=offset,
            source_profile=runtime_inputs.symbolic_forward_source,
            linear_source_profile="logical_group128_fp32", mla_attention_source="flashmla_bf16",
            use_deep_gemm_bmm=False,
            allreduce_fusion=(runtime_inputs.symbolic_forward_source in (SOURCE_FUSED_PROFILE, SOURCE_TARGET_FUSED_PROFILE)),
            fused_shared_experts=1, scale_routed_in_topk=True,
            grouped_topk_source="triton_grouped", moe_runner_source="cutlass_fp8")
        if runtime_inputs.graph_execution.enabled:
            if isinstance(runtime_inputs.model_source_inputs, SGLangEagleChainState):
                state = runtime_inputs.model_source_inputs
                if state.phase == "prefill" and any(request.computed_tokens for request in step.requests):
                    raise NotImplementedError("SGLang EAGLE prefix prefill requires graph-off source inputs")
                if state.phase == "verify":
                    batch = len(step.requests)
                    if state.proposal_depth != 2:
                        raise NotImplementedError("EAGLE graph source requires its topk1/steps2 source")
                    from .traced_sglang import resolve_eagle_capture_batch
                    captured_batch = resolve_eagle_capture_batch(state, runtime_inputs.graph_execution.capture_sizes)
                    arguments["eagle_graph_capture_batch_size"] = captured_batch
                    if captured_batch > batch:
                        # Same reserved FlashMLA page0 as ordinary SG graphs;
                        # translate declared logical arenas, never native pages.
                        arguments["slot_mapping"] = tuple(slot + 64 for slot in slots)
                        arguments["cache_capacity_tokens"] = offset + 64
                        state = replace(state, draft_slot_mapping=tuple(slot + 64 for slot in state.draft_slot_mapping),
                            draft_decode_slot_mapping=tuple(slot + 64 for slot in state.draft_decode_slot_mapping),
                            draft_cache_capacity_tokens=state.draft_cache_capacity_tokens + 64)
                        arguments["eagle_state"] = state
                # Source prefill remains eager; its first draft-extend uses
                # different preparation from the verified three-stage replay.
            else:
                token_shape = runtime_inputs.graph_execution.resolve(step.token_count,
                    uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
                if token_shape.runtime_mode == "FULL":
                    arguments["graph_token_shape"] = token_shape
                    if token_shape.padding_tokens:
                        # SGLang's FlashMLA allocator reserves page0 (64 rows)
                        # for graph dummy slot0. Keep active logical ranges apart;
                        # this is not a native allocator/capacity claim.
                        arguments["slot_mapping"] = tuple(slot + 64 for slot in slots)
                        arguments["cache_capacity_tokens"] = offset + 64
        graph_inputs = runtime_inputs.model_source_inputs if isinstance(runtime_inputs.model_source_inputs, SGLangGraphInputs) else None
        if graph_inputs is not None:
            arguments["graph_inputs"] = graph_inputs
        if runtime_inputs.symbolic_forward_source in (SOURCE_TARGET_PROFILE, SOURCE_TARGET_FUSED_PROFILE):
            source_inputs = (arguments.pop("eagle_state") if "eagle_state" in arguments else
                (None if graph_inputs is not None else runtime_inputs.model_source_inputs))
            if isinstance(source_inputs, SGLangPrefillInputs):
                if source_inputs.request_ids != tuple(request.request_id for request in step.requests):
                    raise ValueError("SGLang prefill metadata must match the scheduled request order")
                arguments.update(one_shot_kv_indices=(source_inputs.one_shot_kv_indices or None),
                    max_kv_chunk_capacity=source_inputs.max_kv_chunk_capacity,
                    prefix_chunk_kv_indices=source_inputs.prefix_chunk_kv_indices)
                source_inputs = None
            return finish(trace_sglang_deepseek_v3_forward(model, step, pool, runtime_inputs.expert_routing,
                **arguments, prefill_source=PREFILL_PREFIX_SOURCE,
                sampling_source=runtime_inputs.target_sampling_source, eagle_state=source_inputs))
        return finish(trace_sglang_deepseek_v3_decode(model, step, pool, runtime_inputs.expert_routing, **arguments))
    if runtime_inputs.symbolic_forward_source in ("vllm_0_27_1_deepseek_v3_sync_fp8", "vllm_0_11_deepseek_v3_sync_fp8"):
        if runtime_inputs.mtp_draft_sampling_source == "vllm_0_27_1_eager_full_logits_greedy":
            from .traced_deepseek_v3 import build_deepseek_v3_mtp_workload
            return build_deepseek_v3_mtp_workload(step, model, deployment,
                launch_policy=launch_policy, runtime_inputs=runtime_inputs,
                kv_cache_events=kv_cache_events, trace_observer=trace_observer)
        from .traced_deepseek_v3 import trace_deepseek_v3_decode
        if deployment.mode != "colocated":
            raise NotImplementedError("DeepSeek-V3 initial virtual source requires colocated deployment")
        if (offset == 0 or max(r.computed_tokens + r.scheduled_tokens for r in step.requests)
                > runtime_inputs.symbolic_source_max_model_len
                or runtime_inputs.symbolic_source_max_model_len > model.max_position_embeddings
                or step.token_count > runtime_inputs.symbolic_source_max_batched_tokens):
            raise ValueError("DeepSeek-V3 declared source constructor limits must contain its scheduled contexts and rows")
        return finish(trace_deepseek_v3_decode(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
            target_sampling_source=runtime_inputs.target_sampling_source,
            source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens))
    if runtime_inputs.symbolic_forward_source == "vllm_0_11_glm4_sync":
        if deployment.mode not in ("colocated", "pd_disagg", "af_disagg", "pd_af_disagg"):
            raise NotImplementedError("GLM4 common forward requires colocated, PD or its configured AF pair")
        has_mtp = any(r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests)
        if has_mtp:
            if runtime_inputs.mtp_draft_sampling_source != "vllm_0_11_eager_full_logits_greedy":
                raise NotImplementedError("GLM4 MTP-off source cannot omit its proposer/rejection work")
            if pool.kv_cache is None:
                raise ValueError("GLM4 MTP requires an explicit cache block-size contract")
            block_size = next(tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm")
            slots, offset = [], 0
            for request in step.requests:
                slots.extend(range(offset + request.computed_tokens,
                    offset + request.computed_tokens + request.scheduled_tokens))
                extent = request.computed_tokens + request.scheduled_tokens + max(request.mtp_proposer_tokens - 1, 0)
                offset += -(-extent // block_size) * block_size
        if deployment.mode in ("af_disagg", "pd_af_disagg"):
            return finish(trace_af_forward(model, step, deployment, runtime_inputs.expert_routing,
                input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
                cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
                max_model_len=runtime_inputs.symbolic_source_max_model_len,
                max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
                max_requests=runtime_inputs.symbolic_source_max_requests))
        if has_mtp:
            return finish(trace_glm4_mtp_forward(model, step, pool, runtime_inputs.expert_routing,
                input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
                cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
                max_model_len=runtime_inputs.symbolic_source_max_model_len,
                max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
                max_requests=runtime_inputs.symbolic_source_max_requests))
        return finish(trace_glm4_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
            # GPUModelRunner._sample selects the ordinary sampler when this
            # pool has no speculative metadata. PD prefill keeps
            # the global rejection declaration but does not run a proposer.
            target_sampling_source=("vllm_0_11_plain_greedy_bf16"
                if runtime_inputs.target_sampling_source == "vllm_0_11_greedy_rejection_bf16"
                else runtime_inputs.target_sampling_source)))
    if runtime_inputs.symbolic_forward_source == "vllm_0_30_v4_sync_fp8_ds_mla":
        if model.model_type != "deepseek_v4":
            raise NotImplementedError("symbolic V4 source requires its accurate DeepSeek-V4 model config")
        if deployment.mode != "colocated":
            raise NotImplementedError("symbolic V4 source currently requires colocated deployment")
        if hardware_by_id is None or pool.hardware_id not in hardware_by_id:
            raise ValueError("symbolic V4 source needs the explicit deployment HWSpec for mHC split storage")
        # This is only a declared logical cache arena containing current
        # request ranges and one descriptor row for the largest compression
        # ratio. It never replaces the physical/native KV capacity ledger.
        alignment = max(model.compress_ratios)
        logical_capacity = -(-max(offset, alignment) // alignment) * alignment
        return finish(trace_deepseek_v4_forward(model, step, pool, hardware_by_id[pool.hardware_id],
            runtime_inputs.expert_routing, input_ids=runtime_inputs.embedding_token_ids,
            positions=tuple(positions), slot_mapping=tuple(slots), cache_capacity_tokens=logical_capacity,
            source_max_model_len=runtime_inputs.v4_source_max_model_len,
            max_batched_tokens=runtime_inputs.v4_source_max_batched_tokens,
            source_use_deep_gemm=runtime_inputs.v4_source_use_deep_gemm,
            source_prompt_lengths=runtime_inputs.v4_source_prompt_lengths,
            indexer_selection_source=runtime_inputs.v4_indexer_selection_source,
            workspace_initial_capacity_bytes=runtime_inputs.symbolic_workspace_initial_capacity_bytes,
            sample_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16",
            cast_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16",
            linear_source_profile="vllm_0_30_row_major_group128_fp32"))
    if (runtime_inputs.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
            and parallel.pipeline_parallel_size > 1):
        from .traced_glm_pipeline import trace_glm_pipeline_forward
        if deployment.mode != "colocated" or len(deployment.worker_pools) != 1:
            raise NotImplementedError("GLM PP source projection requires one colocated pool")
        return finish(trace_glm_pipeline_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            workspace_initial_capacity_bytes=runtime_inputs.symbolic_workspace_initial_capacity_bytes,
            selection_source=runtime_inputs.glm_indexer_selection_source,
            source_profile=runtime_inputs.symbolic_forward_source,
            target_sampling_source=runtime_inputs.target_sampling_source,
            prefill_attention_source=runtime_inputs.glm_prefill_attention_source,
            source_prompt_lengths=runtime_inputs.symbolic_source_prompt_lengths))
    if runtime_inputs.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce",
            "vllm_0_30_k2_6_int4_sync_explicit_allreduce"):
        single_stage = (len(pool.ranks) == parallel.tensor_parallel_size
            and all(value == 1 for value in (parallel.pipeline_parallel_size, parallel.sequence_parallel_size,
                parallel.context_parallel_size, parallel.data_parallel_size)))
        if (deployment.mode not in ("colocated", "pd_disagg")
                or not (single_stage or _kimi_pipeline_source_scope(model, pool))):
            raise NotImplementedError("virtual MLA production dispatch requires colocated/PD, one rank per TP lane and PP/SP/CP/DP1")
    if deployment.mode == "pd_disagg":
        # The same per-pool forward runs on either side of the existing
        # request-scoped KV handoff. No model graph is copied into a PD path.
        from src.serving.deployment.validated_scope import is_validated_tp1_tp2_multi_pool_scope
        from src.serving.deployment.deployment_config import worker_role_phase
        single_rank_source = (model.model_type in ("qwen3", "qwen3_moe", "llama")
            and model.attention_layout == "standard_gqa"
            and {worker_role_phase(item.role) for item in deployment.worker_pools.values()} == {"prefill", "decode"}
            and all(item.hardware_id == "nvidia_a800" and len(item.ranks) == 1
                and item.parallel_plan.config.worker_count == 1 and item.parallel_plan.config.is_rank_local
                and not item.parallel_plan.config.enable_expert_parallel for item in deployment.worker_pools.values()))
        mla_source = (runtime_inputs.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce",
            "vllm_0_30_k2_6_int4_sync_explicit_allreduce") and _ordinary_mla_pd_source_scope(model, deployment))
        if not (single_rank_source or is_validated_tp1_tp2_multi_pool_scope(deployment, model) or mla_source):
            raise NotImplementedError("virtual PD forward requires one-rank GQA, validated TP1/TP2 ownership or an existing H200 MLA per-pool source")
    if deployment.mode in ("af_disagg", "pd_af_disagg"):
        graph_token_shape = None
        if runtime_inputs.graph_execution.enabled:
            uniform_decode = all(request.phase == "decode" and request.scheduled_tokens == 1 for request in step.requests)
            token_shape = runtime_inputs.graph_execution.resolve(step.token_count, uniform_decode=uniform_decode)
            if token_shape.runtime_mode == "FULL" and not uniform_decode:
                raise ValueError("AF FULL source capture requires uniform decode; use FULL_AND_PIECEWISE for prefill or mixed work")
            if token_shape.runtime_mode in ("FULL", "PIECEWISE"):
                graph_token_shape = token_shape
        return finish(trace_af_forward(model, step, deployment, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids,
            positions=tuple(positions), slot_mapping=tuple(slots), cache_capacity_tokens=offset,
            source_profile=runtime_inputs.symbolic_forward_source,
            model_execution_version=runtime_inputs.model_execution_version,
            graph_token_shape=graph_token_shape,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if graph_token_shape is not None and graph_token_shape.runtime_mode == "PIECEWISE" else ())))
    if runtime_inputs.symbolic_forward_source == "vllm_0_30_k2_6_int4_sync_explicit_allreduce":
        if parallel.pipeline_parallel_size > 1 and (deployment.mode != "colocated" or len(deployment.worker_pools) != 1):
            raise NotImplementedError("Kimi pipeline source requires one colocated pool")
        token_shape = (runtime_inputs.graph_execution.resolve(step.token_count,
            uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
            if runtime_inputs.graph_execution.enabled else None)
        captured = token_shape is not None and token_shape.runtime_mode in ("FULL", "PIECEWISE")
        return finish(trace_kimi_k2_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            source_profile=runtime_inputs.symbolic_forward_source, target_sampling_source=runtime_inputs.target_sampling_source,
            include_lm_head=True, source_prompt_lengths=runtime_inputs.symbolic_source_prompt_lengths,
            graph_token_shape=token_shape if captured else None,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if captured and token_shape.runtime_mode == "PIECEWISE" else ()),
            source_block_size=None if pool.kv_cache is None else next(
                tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm")))
    if parallel.pipeline_parallel_size > 1:
        if (deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                or runtime_inputs.symbolic_forward_source not in ("cuda_eager_inplace", "vllm_ascend_eager")):
            raise NotImplementedError("virtual PP requires one colocated dense source pool")
        token_shape = (runtime_inputs.graph_execution.resolve(step.token_count,
            uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
            if runtime_inputs.graph_execution.enabled else None)
        captured = token_shape is not None and token_shape.runtime_mode in ("FULL", "PIECEWISE")
        return finish(trace_dense_pipeline_forward(model, step, pool,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
            model_execution_version=runtime_inputs.model_execution_version,
            graph_token_shape=token_shape if captured else None,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if captured and token_shape.runtime_mode == "PIECEWISE" else ())))
    # MLA callers validate their own precise checkpoint, TP/EP and source
    # metadata domain before the generic dense/Qwen-only EP gate below.
    if runtime_inputs.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce"):
        if runtime_inputs.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy":
            from .traced_glm_mtp import trace_glm_dsa_mtp_forward
            if deployment.mode != "colocated":
                raise NotImplementedError("V32 padded MTP source currently requires colocated logical arenas")
            return finish(trace_glm_dsa_mtp_forward(model, step, pool, runtime_inputs.expert_routing,
                input_ids=runtime_inputs.embedding_token_ids, source_profile=runtime_inputs.symbolic_forward_source,
                source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
                max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
                max_requests=runtime_inputs.symbolic_source_max_requests,
                workspace_initial_capacity_bytes=runtime_inputs.symbolic_workspace_initial_capacity_bytes,
                selection_source=runtime_inputs.glm_indexer_selection_source,
                prefill_attention_source=runtime_inputs.glm_prefill_attention_source,
                source_inputs=runtime_inputs.model_source_inputs))
        return finish(trace_glm_dsa_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            workspace_initial_capacity_bytes=runtime_inputs.symbolic_workspace_initial_capacity_bytes,
            selection_source=runtime_inputs.glm_indexer_selection_source,
            prefill_attention_source=runtime_inputs.glm_prefill_attention_source,
            source_prompt_lengths=runtime_inputs.symbolic_source_prompt_lengths,
            source_profile=runtime_inputs.symbolic_forward_source, target_sampling_source=runtime_inputs.target_sampling_source))
    if (deployment.mode not in ("colocated", "pd_disagg") or pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
            or len(pool.ranks) != parallel.tensor_parallel_size
            or (parallel.enable_expert_parallel and model.model_type != "qwen3_moe")
            or any(value != 1 for value in (parallel.pipeline_parallel_size, parallel.sequence_parallel_size,
                parallel.context_parallel_size, parallel.data_parallel_size))):
        raise NotImplementedError("symbolic dense production path requires its CUDA-alike colocated source, PP/SP/CP/DP1 EP-off")
    if model.model_type == "qwen3_moe":
        graph_token_shape = None
        if runtime_inputs.graph_execution.enabled:
            token_shape = runtime_inputs.graph_execution.resolve(step.token_count,
                uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
            if token_shape.runtime_mode in ("FULL", "PIECEWISE"):
                graph_token_shape = token_shape
        arguments = dict(
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
            model_execution_version=runtime_inputs.model_execution_version,
            sample_logits=runtime_inputs.target_sampling_source not in (None, "vllm_0_30_plain_sampler_bf16"),
            cast_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16",
            graph_token_shape=graph_token_shape,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if graph_token_shape is not None and graph_token_shape.runtime_mode == "PIECEWISE" else ()))
        ordinary_program = (source_program_cache is not None and deployment.mode == "colocated" and not host_transfers
            and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE,
                KVCacheAction.HIT, KVCacheAction.MISS) for event in kv_cache_events))
        return finish(_moe_parallel_source_program(model, step, pool, runtime_inputs.expert_routing, arguments,
            cache=source_program_cache, launch_policy=launch_policy) if ordinary_program else
            trace_qwen3_moe_forward(model, step, pool, runtime_inputs.expert_routing, **arguments))
    if parallel.tensor_parallel_size > 1:
        graph_token_shape = None
        if runtime_inputs.graph_execution.enabled:
            token_shape = runtime_inputs.graph_execution.resolve(step.token_count,
                uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
            if token_shape.runtime_mode in ("FULL", "PIECEWISE"):
                graph_token_shape = token_shape
        arguments = dict(input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions),
            slot_mapping=tuple(slots), cache_capacity_tokens=offset,
            source_profile=runtime_inputs.symbolic_forward_source, graph_token_shape=graph_token_shape,
            model_execution_version=runtime_inputs.model_execution_version,
            sample_logits=runtime_inputs.target_sampling_source not in (None, "vllm_0_30_plain_sampler_bf16"),
            cast_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16",
            graph_inputs=dense_graph_inputs,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if graph_token_shape is not None and graph_token_shape.runtime_mode == "PIECEWISE" else ()))
        ordinary_program = (source_program_cache is not None and deployment.mode == "colocated"
            and pool.role == "colocated" and not host_transfers
            and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE,
                KVCacheAction.HIT, KVCacheAction.MISS) for event in kv_cache_events))
        return finish(_dense_parallel_source_program(model, step, pool, arguments,
            cache=source_program_cache, launch_policy=launch_policy) if ordinary_program else
            trace_dense_parallel_forward(model, step, pool, **arguments))
    if model.model_type == "qwen3":
        forward = trace_qwen3_forward
    elif model.model_type == "llama":
        forward = trace_llama_forward
    else:
        raise NotImplementedError("symbolic dense source requires an adapted Qwen3 or Llama forward")
    if runtime_inputs.graph_execution.enabled:
        token_shape = runtime_inputs.graph_execution.resolve(step.token_count,
            uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
        if token_shape.runtime_mode in ("FULL", "PIECEWISE"):
            (_validate_qwen3_source if model.model_type == "qwen3" else _validate_llama_source)(model)
            arguments = dict(input_ids=runtime_inputs.embedding_token_ids,
                positions=tuple(positions), slot_mapping=tuple(slots), cache_capacity_tokens=offset,
                source_profile=runtime_inputs.symbolic_forward_source, rank=pool.ranks[0], namespace=step.step_id,
                model_type=Qwen3Model if model.model_type == "qwen3" else LlamaModel,
                hardware_id=pool.hardware_id, rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type),
                graph_token_shape=token_shape,
                model_execution_version=runtime_inputs.model_execution_version,
                maca_copy_runtime_source=runtime_inputs.maca_copy_runtime_source,
                graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                    if token_shape.runtime_mode == "PIECEWISE" else ()))
            if (pool.hardware_id == "ascend_910b" and runtime_inputs.model_execution_version == "vllm_0_18_0"
                    and runtime_inputs.graph_execution.configured_mode == "PIECEWISE"):
                arguments["ascend_piecewise_source"] = True
            ordinary_program = (source_program_cache is not None and deployment.mode == "colocated"
                and parallel.is_rank_local and not parallel.enable_expert_parallel and not host_transfers
                and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE,
                    KVCacheAction.HIT, KVCacheAction.MISS) for event in kv_cache_events))
            return finish(_dense_source_program(model, step, arguments, cache=source_program_cache,
                launch_policy=launch_policy) if ordinary_program else _trace_dense_forward(model, step, **arguments))
    return finish(forward(model, step, input_ids=runtime_inputs.embedding_token_ids,
        positions=tuple(positions), slot_mapping=tuple(slots), cache_capacity_tokens=offset,
        source_profile=runtime_inputs.symbolic_forward_source, rank=pool.ranks[0],
        namespace=step.step_id, hardware_id=pool.hardware_id,
        model_execution_version=runtime_inputs.model_execution_version,
        maca_copy_runtime_source=runtime_inputs.maca_copy_runtime_source,
        sample_logits=runtime_inputs.target_sampling_source not in (None, "vllm_0_30_plain_sampler_bf16"),
        cast_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16",
        **({"ascend_piecewise_source": True} if pool.hardware_id == "ascend_910b"
            and runtime_inputs.model_execution_version == "vllm_0_18_0"
            and runtime_inputs.graph_execution.configured_mode == "PIECEWISE" else {})))


# Only import-time known source may bind metadata without executing its prefix.
_DENSE_METADATA_SOURCE_CONTRACT = _DenseMetadataSourceGuard()
_DENSE_METADATA_SOURCE_CONTRACT.seal()
