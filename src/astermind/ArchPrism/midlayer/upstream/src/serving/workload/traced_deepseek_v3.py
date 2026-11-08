"""Versioned DeepSeek-V3 forwards on shared virtual Tensor operators.

The0.27.1 profile binds upstream/tags/v0.27.1 deepseek_v2.py and mla.py.
The0.11 TP1 profile binds the pinned native Scheduler installation's
model_executor/models/deepseek_v2.py, layers/mla.py and v1/attention/backends/
mla/common.py: Q projection precedes KV normalization, K is concatenated,
and context output is merged into fresh storage before copying to the caller.
Both retain explicit source identity and reuse the existing FP8/MLA useful
mathematical payloads. Provider selection, workspace and physical capacity
remain independent; neither profile claims a native GPU execution.
"""

from contextlib import contextmanager
from dataclasses import replace

from .traced_forward import ForwardCollectiveGroup, ForwardTrace
from .traced_models import (
    DeepseekV2MLAInput, DeepseekV2MLADecodeAttention, DeepseekV2MLP,
    DeepseekV2KimiMoERunner, Qwen3DecoderLayer, Qwen3Model,
    DeepseekV2MLAFreshPrefillAttention, DeepseekV2MLAMixedAttention,
    MLAFreshPrefillBackend, MLAContextPrefillBackend,
    _source_expert_resolutions, _source_linear,
)
from .traced_operators import (
    SymbolicAdd, SymbolicBatchedLinear, SymbolicConcat, SymbolicLinear,
    SymbolicEmbedding, SymbolicGreedy, SymbolicLogitsRowSelect, SymbolicMLACacheUpdate, SymbolicMLALatentDecode,
    SymbolicOperator, SymbolicPositionZeroMask, SymbolicRMSNorm, SymbolicRotary, SymbolicRowParallelLinear,
    SymbolicScalarMultiply, SymbolicTPLogitsProcessor, SymbolicVocabParallelEmbedding,
)
from .workload_graph import WorkloadKind


SOURCE_PROFILE = "vllm_0_27_1_deepseek_v3_sync_fp8"
SOURCE_PROFILE_011 = "vllm_0_11_deepseek_v3_sync_fp8"


class DeepseekV3Rotary(SymbolicRotary):
    """Interleaved YaRN uses the precomputed cache; no cache values computed."""

    def __call__(self, positions, query, key):
        rows = query.shape[0]
        node = self.node(WorkloadKind.ROTARY, {"elements": rows * (self.heads + 1) * self.head_dim,
            "seq_len": rows, "num_query_heads": self.heads, "num_kv_heads": 1,
            "head_dim": self.head_dim}, "bf16", "rotary", {"key_present": True, "is_neox": False})
        self.trace.call(node, (positions, query, key, self.cache), shape=query.shape, dtype="bf16", out=query, writes=(key,))
        return query, key


class DeepseekV3MLAInput(DeepseekV2MLAInput):
    """Reuse the MLA wrapper forward, with V3's exact source constructors."""

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, rotary_cache):
        self.trace, self.model, self.heads = trace, model, model.num_attention_heads // tp_size
        path = f"model.layers.{layer_index}.self_attn"
        self.fused_qkv_a_proj = _source_linear(trace, model, pool, prefix + ":qkv-a", model.hidden_size,
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim, path + ".fused_qkv_a_proj")
        self.q_a_layernorm = SymbolicRMSNorm(trace, prefix + ":q-a-norm", pool,
            model.q_lora_rank, model.target_semantics.rms_norm_eps)
        self.kv_a_layernorm = SymbolicRMSNorm(trace, prefix + ":kv-a-norm", pool,
            model.kv_lora_rank, model.target_semantics.rms_norm_eps)
        self.q_b_proj = _source_linear(trace, model, pool, prefix + ":q-b", model.q_lora_rank,
            self.heads * (model.qk_nope_head_dim + model.qk_rope_head_dim), path + ".q_b_proj")
        self.rotary_emb = DeepseekV3Rotary(trace, prefix + ":rope", pool,
            self.heads, 1, model.qk_rope_head_dim, rotary_cache)


class DeepseekV3MLAInput011(DeepseekV3MLAInput):
    """Pinned0.11 layers/mla.py forward_native ordering, no fused RMSNorm."""

    @contextmanager
    def __call__(self, positions, hidden_states):
        model, rows = self.model, hidden_states.shape[0]
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split((model.q_lora_rank, model.kv_lora_rank + model.qk_rope_head_dim))
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]
        kv_c, k_pe = kv_lora.split((model.kv_lora_rank, model.qk_rope_head_dim))
        latent = self.kv_a_layernorm(kv_c)
        q = q.view_as_shape(rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
        k_pe = k_pe.unsqueeze(1)
        _, k_pe = self.rotary_emb(positions, q.narrow(-1, model.qk_nope_head_dim, model.qk_rope_head_dim), k_pe)
        yield q, latent, k_pe


class DeepseekV3PrefillBackend011(MLAFreshPrefillBackend):
    """0.11 MLACommonImpl expands latent KV then torch.cat materializes K."""

    def _expand_kv(self, latent, rope):
        model, rows = self.model, latent.shape[0]
        kv = self.kv_b_proj(latent)[0].view_as_shape(rows, self.heads, model.qk_nope_head_dim + model.v_head_dim)
        nope, value = kv.split((model.qk_nope_head_dim, model.v_head_dim))
        key = SymbolicConcat(self.trace, self.prefix + ":concat-key", self.pool)(
            nope, rope.expand(rows, self.heads, model.qk_rope_head_dim))
        return key, value


class DeepseekV3ContextBackend011(MLAContextPrefillBackend, DeepseekV3PrefillBackend011):
    """0.11 merges O/LSE into fresh output, then copies to caller output."""

    def __call__(self, query, latent, rope, output):
        fresh = self.trace.tensor(self.trace.unique_name(self.prefix + ":context-merged-output"), output.shape, "bf16")
        super().__call__(query, latent, rope, fresh)
        self.copy_output(fresh, output)


class DeepseekV3DecodeAttention(DeepseekV2MLADecodeAttention):
    """Reuse absorption, cache append, latent attention and output forward."""

    def __init__(self, trace, model, pool, prefix, *, layer_index, group, collective,
                 kv_lens, cache, slots, slot_tensor, rotary_cache, q_lens=None,
                 attention_source=None, source_batch_invariant=None, source_version="vllm_0_27_1"):
        self.trace, self.model, self.name = trace, model, prefix
        self.graph_token_shape = None
        input_type = DeepseekV3MLAInput011 if source_version == "vllm_0_11_0" else DeepseekV3MLAInput
        self.inputs = input_type(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=1 if group is None else len(group.traces), rotary_cache=rotary_cache)
        self.heads = self.inputs.heads
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_tensor,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        self.q_absorb = SymbolicBatchedLinear(trace, prefix + ":q-absorb", pool,
            self.heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.concat = SymbolicConcat(trace, prefix + ":concat-query", pool)
        self.q_lens = (1,) * len(kv_lens) if q_lens is None else q_lens
        multiquery = any(q > 1 for q in self.q_lens)
        options = ({"q_lens": self.q_lens, "source_profile": attention_source,
            "source_batch_invariant": source_batch_invariant} if multiquery else {})
        self.latent = SymbolicMLALatentDecode(trace, prefix + ":latent", pool,
            heads=self.heads, kv_lens=kv_lens, cache=cache, **options)
        self.v_up = SymbolicBatchedLinear(trace, prefix + ":v-up", pool, self.heads, model.kv_lora_rank, model.v_head_dim)
        projection = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.heads * model.v_head_dim, model.hidden_size, f"model.layers.{layer_index}.self_attn.o_proj")
        self.o_proj = projection if group is None else SymbolicRowParallelLinear(projection, group, collective)

    def __call__(self, positions, hidden_states):
        rows = sum(self.q_lens)
        if (hidden_states.shape != (rows, self.model.hidden_size) or hidden_states.dtype != "bf16"
                or positions.shape != (rows,) or positions.dtype != "int64"):
            raise ValueError("V3 MLA decode requires all declared BF16 query rows and INT64 positions")
        with self.inputs(positions, hidden_states) as (query, latent, rope):
            return self.o_proj(self._mla(query, latent, rope))[0]


class DeepseekV3PrefillAttention(DeepseekV2MLAFreshPrefillAttention):
    """Versioned MLA prefill, BF16/no fused output quant.

    Reuse the mathematical body after constructing the official V3 FP8
    projections. The source returns a fresh backend output, then copies it
    into the caller slice; context additionally gathers and merges O/LSE.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, group, collective,
                 q_lens, cache, slots, slot_tensor, rotary_cache, context_chunk_rows=None,
                 context_lengths=(), context_offsets=(), source_version="vllm_0_27_1"):
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.q_lens, self.cache = q_lens, cache
        input_type = DeepseekV3MLAInput011 if source_version == "vllm_0_11_0" else DeepseekV3MLAInput
        self.inputs = input_type(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=1 if group is None else len(group.traces), rotary_cache=rotary_cache)
        self.heads = self.inputs.heads
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_tensor,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        path = f"model.layers.{layer_index}.self_attn"
        if bool(context_lengths) != bool(context_offsets):
            raise ValueError("V3 batched contexts require matching lengths and offsets")
        backend = ((DeepseekV3ContextBackend011 if context_lengths else DeepseekV3PrefillBackend011)
            if source_version == "vllm_0_11_0" else
            (MLAContextPrefillBackend if context_lengths else MLAFreshPrefillBackend))
        context = {"context_lengths": context_lengths, "context_offsets": context_offsets,
            "context_chunk_rows": context_chunk_rows} if context_lengths else {}
        self.prefill_backend = backend(trace, model, pool, prefix,
            heads=self.heads, q_lens=q_lens, cache=cache, module_path=path, **context)
        projection = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.heads * model.v_head_dim, model.hidden_size, path + ".o_proj")
        self.o_proj = projection if group is None else SymbolicRowParallelLinear(projection, group, collective)


class DeepseekV3MixedAttention(DeepseekV2MLAMixedAttention):
    """Ordinary MLA decode prefix and prefill suffix share QKV/RoPE."""

    def __init__(self, trace, model, pool, prefix, *, layer_index, group, collective,
                 q_lens, kv_lens, num_decodes, cache, slots, slot_tensor, rotary_cache, context_chunk_rows=None, source_version="vllm_0_27_1"):
        self.trace, self.model, self.prefix = trace, model, prefix
        self.rows, self.num_decodes = sum(q_lens), num_decodes
        self.decode_rows = sum(q_lens[:num_decodes])
        self.graph_token_shape = None
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_tensor,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        count = self.rows - num_decodes
        context_lengths = tuple(k - q for q, k in zip(q_lens[num_decodes:], kv_lens[num_decodes:]))
        offsets, cursor = [], num_decodes
        for q, context in zip(q_lens[num_decodes:], context_lengths):
            offsets.append(slots[cursor] - context)
            cursor += q
        if any(context_lengths) and any(length <= 0 for length in context_lengths):
            raise NotImplementedError("V3 batched context source requires every prefill request to have a context")
        self.prefill = DeepseekV3PrefillAttention(trace, model, pool, prefix + ":prefill",
            layer_index=layer_index, group=group, collective=collective,
            q_lens=q_lens[num_decodes:], cache=cache, slots=slots[num_decodes:],
            slot_tensor=slot_tensor.narrow(0, num_decodes, count), rotary_cache=rotary_cache,
            context_lengths=context_lengths if any(context_lengths) else (),
            context_offsets=tuple(offsets) if any(context_lengths) else (), context_chunk_rows=context_chunk_rows,
            source_version=source_version)
        self.decode = None
        if num_decodes:
            self.decode = DeepseekV3DecodeAttention(trace, model, pool, prefix + ":decode",
                layer_index=layer_index, group=group, collective=collective, kv_lens=kv_lens[:num_decodes],
                cache=cache, slots=slots[:num_decodes], slot_tensor=slot_tensor.narrow(0, 0, num_decodes),
                rotary_cache=rotary_cache, source_version=source_version)


class DeepseekV3GroupedRouting(SymbolicOperator):
    def __init__(self, trace, name, pool, model, *, source_version="vllm_0_27_1"):
        super().__init__(trace, name, pool)
        self.model = model
        self.source_prefix = {"vllm_0_11_0": "vllm_0_11", "vllm_0_27_1": "vllm_0_27_1"}[source_version]
        self.bias = trace.tensor(name + ":correction-bias", (model.num_experts,), "fp32", persistent=True)

    def __call__(self, hidden, logits):
        model, rows = self.model, hidden.shape[0]
        if hidden.dtype != "bf16" or logits.dtype != "bf16" or logits.shape != (rows, model.num_experts):
            raise ValueError("DeepSeek-V3 routing requires its BF16 gate logits")
        node = self.node(WorkloadKind.VECTOR, {"tokens": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
            "intermediate": model.moe_intermediate_size}, "bf16", "moe_topk", {
                "source_component": self.source_prefix + "_symbolic_grouped_routing",
                "source_program": self.source_prefix + "_bf16_sigmoid_bias_grouped_topk",
                "mathematical_program": "bf16_sigmoid_bias_grouped_topk_flops_bytes_v1",
                "normalize_topk_probability": model.target_semantics.moe_normalize_topk_probability,
                "expert_group_count": model.target_semantics.moe_group_count,
                "topk_group_count": model.target_semantics.moe_topk_group_count,
                "routed_scaling_factor": 1.0})
        return self.trace.call_outputs(node, (logits, self.bias), outputs=(
            ("weights", (rows, model.num_experts_per_tok), "fp32"),
            ("ids", (rows, model.num_experts_per_tok), "int32")))


class DeepseekV3Experts(SymbolicOperator):
    def __init__(self, trace, name, pool, model, *, resolution, invocation_id, tp_size=8):
        super().__init__(trace, name, pool)
        self.model, self.resolution, self.invocation = model, resolution, invocation_id
        if type(tp_size) is not int or tp_size not in (1, 8):
            raise ValueError("V3 expert widths require their declared TP1/8 source")
        self.tp_size = tp_size

    def __call__(self, hidden, weights, ids):
        model, rows = self.model, hidden.shape[0]
        counts = self.resolution.tokens_per_expert
        if (hidden.shape != (rows, model.hidden_size) or hidden.dtype != "bf16"
                or weights.shape != (rows, model.num_experts_per_tok) or weights.dtype != "fp32"
                or ids.shape != weights.shape or ids.dtype != "int32"
                or len(counts) != model.num_experts or sum(counts) != rows * model.num_experts_per_tok):
            raise ValueError("DeepSeek-V3 experts require conserved declared routing and typed ports")
        node = self.node(WorkloadKind.FUSED_MOE, {"tokens": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
            "intermediate": model.moe_intermediate_size // self.tp_size}, "bf16", "full", {
                "source_component": "symbolic_dsv3_routed_experts", "expert_parallel_size": 1,
                "expert_rank": 0, "owned_experts": tuple(range(model.num_experts)),
                "tokens_per_expert": counts, "local_tokens_per_expert": counts,
                "expert_placement_strategy": "linear", "moe_invocation_id": self.invocation,
                "output_is_reduced": False, "routed_scaling_factor": 1.0,
                "routed_weight_dtype": "fp8", "checkpoint_weight_dtype": "fp8", "weight_group_size": 0,
                "routing_source_stage": "moe_topk", "routing_provenance": dict(self.resolution.routing_provenance),
                "symbolic_workspace_complete": False})
        return self.trace.call(node, (hidden, weights, ids), shape=hidden.shape, dtype="bf16")


class DeepseekV3MoERunner(DeepseekV2KimiMoERunner):
    """Same modular Runner body: scale routed output, add shared, TP sum."""

    def __init__(self, trace, model, pool, prefix, group, collective, *, layer, resolution, invocation, source_version="vllm_0_27_1"):
        tp_size = 1 if group is None else len(group.traces)
        self.expert_parallel_size, self.invocation_id = 1, invocation
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size,
            model.num_experts, kind=WorkloadKind.ROUTER)
        self.shared_experts = DeepseekV2MLP(trace, model, pool, prefix + ":shared", tp_size=tp_size,
            intermediate_size=model.moe_intermediate_size, module_path=f"model.layers.{layer}.mlp.shared_experts",
            reduce_results=False)
        self.router = DeepseekV3GroupedRouting(trace, prefix + ":route", pool, model, source_version=source_version)
        self.routed_experts = DeepseekV3Experts(trace, prefix + ":routed", pool, model,
            resolution=resolution, invocation_id=invocation, tp_size=tp_size)
        self.output_scale = SymbolicScalarMultiply(trace, prefix + ":output-scale", pool, model.routed_scaling_factor)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)
        self.group, self.config, self.name, self.pool = group, collective, prefix, pool

    def __call__(self, hidden):
        return super().__call__(hidden_states=hidden, router_logits=hidden)


class DeepseekV3Decoder(Qwen3DecoderLayer):
    def __init__(self, trace, model, pool, prefix, *, attention, mlp):
        self.remote_ffn = None
        self.input_layernorm = SymbolicRMSNorm(trace, prefix + ":input-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        self.post_attention_layernorm = SymbolicRMSNorm(trace, prefix + ":post-attn-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        self.self_attn, self.mlp = attention, mlp


class DeepseekV3Model(Qwen3Model):
    def __init__(self, trace, model, pool, group, collective, *, input_ids, kv_lens, slots,
                 cache_capacity_tokens, resolutions, invocations, q_lens, num_decodes,
                 attention_source=None, source_batch_invariant=None, context_chunk_rows=None, source_version="vllm_0_27_1"):
        self.first_stage = self.last_stage = True
        tp_size = 1 if group is None else len(group.traces)
        self.embed_tokens = (SymbolicEmbedding(trace, "embedding", pool, model.hidden_size,
            "bf16", vocab_size=model.vocab_size, model_execution_version=source_version) if group is None
            else SymbolicVocabParallelEmbedding(trace, "embedding", pool, model.hidden_size,
                model.vocab_size, group, collective, token_ids=input_ids))
        self.rope_cache = trace.tensor("model.rope-cache", (model.max_position_embeddings, model.qk_rope_head_dim),
            "bf16", persistent=True)
        slot_tensor = trace.tensor("slot_mapping", (len(slots),), "int64", persistent=True)
        self.layers = []
        for layer in range(model.num_hidden_layers):
            prefix = f"layer-{layer}"
            cache = trace.tensor(prefix + ":kv-cache", (cache_capacity_tokens,
                model.kv_lora_rank + model.qk_rope_head_dim), "bf16", persistent=True)
            if num_decodes < len(q_lens):
                attention = DeepseekV3MixedAttention(trace, model, pool, prefix + ":attention", layer_index=layer,
                    group=group, collective=collective, q_lens=q_lens, kv_lens=kv_lens, num_decodes=num_decodes,
                    cache=cache, slots=slots, slot_tensor=slot_tensor, rotary_cache=self.rope_cache,
                    context_chunk_rows=context_chunk_rows, source_version=source_version)
            else:
                attention = DeepseekV3DecodeAttention(trace, model, pool, prefix + ":attention", layer_index=layer,
                    group=group, collective=collective, kv_lens=kv_lens, cache=cache, slots=slots,
                    slot_tensor=slot_tensor, rotary_cache=self.rope_cache, q_lens=q_lens,
                    attention_source=attention_source, source_batch_invariant=source_batch_invariant, source_version=source_version)
            if model.layer_uses_moe(layer):
                mlp = DeepseekV3MoERunner(trace, model, pool, prefix + ":mlp", group, collective,
                    layer=layer, resolution=resolutions[layer], invocation=invocations[layer], source_version=source_version)
            else:
                mlp = DeepseekV2MLP(trace, model, pool, prefix + ":mlp", intermediate_size=model.intermediate_size,
                    tp_size=tp_size, module_path=f"model.layers.{layer}.mlp", reduce_results=group is not None,
                    reduction_group=group, reduction_config=collective)
            self.layers.append(DeepseekV3Decoder(trace, model, pool, prefix, attention=attention, mlp=mlp))
        self.norm = SymbolicRMSNorm(trace, "final-norm", pool, model.hidden_size, model.target_semantics.rms_norm_eps)


class DeepseekV3MTPPredictorLayer:
    """Actual0.27.1 predictor, including all rows of its first draft pass.

    The caller owns shifted draft IDs, previous target/recycle hidden states,
    the routing resolution and sampling. This component does not implement
    target rejection, proposer input preparation or a complete MTP request.
    Source deepseek_mtp.py:103-134 returns pre-norm logits hidden plus a
    separately normalized recycle; compute_logits applies another head norm
    to the pre-norm input. No target IDs are substituted for draft IDs.
    Multiquery calls explicitly select causal Triton MLA with threshold one,
    or uniform BF16 FlashMLA with threshold128 and BATCH_INVARIANT=False.
    These source branches do not bind an installed attention provider.
    """

    def __init__(self, trace, model, pool, prefix, *, group, collective, gather,
                 input_token_ids, q_lens, kv_lens, slots, cache_capacity_tokens,
                 resolution, invocation_id, source_max_model_len=None, max_batched_tokens=None,
                 attention_source=None, source_batch_invariant=None, cache=None, rotary_cache=None, slot_tensor=None):
        if (model.model_id != "deepseek-ai/DeepSeek-V3" or model.model_type != "deepseek_v3"
                or model.num_hidden_layers != 61 or model.mtp_architecture != "deepseek_mtp"
                or model.num_mtp_layers != 1 or not model.mtp_layer_uses_moe(0)
                or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
                or len(group.traces) != 8):
            raise NotImplementedError("V3 draft body requires official0.27.1 FP8 single-MTP-layer TP8 source")
        if (not isinstance(q_lens, tuple) or not q_lens or any(type(q) is not int or q < 1 for q in q_lens)
                or not isinstance(kv_lens, tuple) or len(kv_lens) != len(q_lens)
                or any(type(k) is not int or not q <= k <= model.max_position_embeddings
                    for q, k in zip(q_lens, kv_lens))):
            raise ValueError("V3 draft body requires positive source query lengths and complete logical contexts")
        rows = sum(q_lens)
        num_decodes = next((index for index, q in enumerate(q_lens) if q > 1), len(q_lens))
        flashmla = attention_source == "vllm_0_27_1_causal_flashmla_bf16_source"
        if attention_source not in (None, "vllm_0_27_1_causal_triton_mla_source", "vllm_0_27_1_causal_flashmla_bf16_source"):
            raise NotImplementedError("V3 draft attention requires the declared0.27.1 causal source branch")
        if flashmla:
            if source_batch_invariant is not False:
                raise NotImplementedError("V3 FlashMLA source requires explicit BATCH_INVARIANT=False")
            if len(set(q_lens)) != 1 or max(q_lens) > 128:
                raise NotImplementedError("V3 FlashMLA source requires uniform queries<=128")
            num_decodes = len(q_lens)
        elif source_batch_invariant is not None:
            raise ValueError("V3 BATCH_INVARIANT declaration requires the explicit FlashMLA source")
        if num_decodes < len(q_lens) and attention_source is None:
            raise NotImplementedError("V3 multiquery draft body requires explicit causal Triton MLA or FlashMLA source selection")
        if any(q == 1 for q in q_lens[num_decodes:]):
            raise NotImplementedError("V3 draft body requires source decode-prefix query order")
        if source_max_model_len is not None and (type(source_max_model_len) is not int
                or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings):
            raise ValueError("V3 draft source maximum length must contain every context")
        if max_batched_tokens is not None and (type(max_batched_tokens) is not int or max_batched_tokens < rows):
            raise ValueError("V3 draft source batch limit must contain every query row")
        if any(q > 1 for q in q_lens) and (source_max_model_len is None or max_batched_tokens is None):
            raise NotImplementedError("V3 multiquery draft body requires explicit source context and batch limits")
        if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
                or not isinstance(slots, tuple) or len(slots) != rows or len(set(slots)) != rows
                or any(type(s) is not int or not 0 <= s < cache_capacity_tokens for s in slots)):
            raise ValueError("V3 draft cache requires distinct declared logical slots and complete context capacity")
        if not flashmla and any(q > 1 and k > q for q, k in zip(q_lens, kv_lens)):
            contexts = tuple(k - q for q, k in zip(q_lens[num_decodes:], kv_lens[num_decodes:]))
            if sum(contexts) > min(8 * source_max_model_len, 65536):
                raise NotImplementedError("V3 draft context exceeds the guaranteed single-chunk source workspace")
            starts, cursor = [], 0
            for q, k in zip(q_lens, kv_lens):
                starts.append(slots[cursor] - (k - q))
                cursor += q
            if (any(start < 0 for start in starts)
                    or any(start + k > following for start, k, following in zip(starts, kv_lens, starts[1:]))
                    or slots != tuple(slot for start, q, k in zip(starts, q_lens, kv_lens)
                        for slot in range(start + k - q, start + k))):
                raise ValueError("V3 draft context requires disjoint logical cache coordinates")
        self.trace, self.model, self.rows, self.kv_lens = trace, model, rows, kv_lens
        self.position_values = tuple(position for q, k in zip(q_lens, kv_lens) for position in range(k - q, k))
        self.embed_tokens = SymbolicVocabParallelEmbedding(trace, prefix + ":embedding", pool,
            model.hidden_size, model.vocab_size, group, collective, token_ids=input_token_ids)
        if len(input_token_ids) != rows:
            raise ValueError("V3 draft token IDs must cover its own draft rows")
        self.position_mask = SymbolicPositionZeroMask(trace, prefix + ":position-mask", pool)
        eps = model.target_semantics.rms_norm_eps
        self.enorm = SymbolicRMSNorm(trace, prefix + ":embedding-norm", pool, model.hidden_size, eps)
        self.hnorm = SymbolicRMSNorm(trace, prefix + ":hidden-norm", pool, model.hidden_size, eps)
        self.concat = SymbolicConcat(trace, prefix + ":concat", pool)
        # The source uses plain torch.nn.Linear, without the checkpoint's
        # quant_config, for the embedding/hidden merge projection.
        self.eh_proj = SymbolicLinear(trace, prefix + ":eh-proj", pool, 2 * model.hidden_size, model.hidden_size)
        if rotary_cache is None:
            rotary_cache = trace.tensor(prefix + ":rope-cache",
                (model.max_position_embeddings, model.qk_rope_head_dim), "bf16", persistent=True)
        if cache is None:
            cache = trace.tensor(prefix + ":kv-cache", (cache_capacity_tokens,
                model.kv_lora_rank + model.qk_rope_head_dim), "bf16", persistent=True)
        if slot_tensor is None:
            slot_tensor = trace.tensor(prefix + ":slot-mapping", (rows,), "int64", persistent=True)
        for value in (rotary_cache, cache, slot_tensor):
            trace._check_tensor(value)
        if (cache.shape != (cache_capacity_tokens, model.kv_lora_rank + model.qk_rope_head_dim)
                or cache.dtype != "bf16" or slot_tensor.shape != (rows,) or slot_tensor.dtype != "int64"
                or rotary_cache.shape != (model.max_position_embeddings, model.qk_rope_head_dim)
                or rotary_cache.dtype != "bf16"):
            raise ValueError("V3 recurrent draft requires the same typed logical KV/rotary storage and current slots")
        self.cache, self.rotary_cache = cache, rotary_cache
        if num_decodes < len(q_lens):
            attention = DeepseekV3MixedAttention(trace, model, pool, prefix + ":attention",
                layer_index=model.num_hidden_layers, group=group, collective=collective,
                q_lens=q_lens, kv_lens=kv_lens, num_decodes=num_decodes,
                cache=cache, slots=slots, slot_tensor=slot_tensor, rotary_cache=rotary_cache,
                context_chunk_rows=None if source_max_model_len is None else min(8 * source_max_model_len, 65536))
        else:
            attention = DeepseekV3DecodeAttention(trace, model, pool, prefix + ":attention",
                layer_index=model.num_hidden_layers, group=group, collective=collective,
                kv_lens=kv_lens, cache=cache, slots=slots, slot_tensor=slot_tensor, rotary_cache=rotary_cache,
                q_lens=q_lens, attention_source=attention_source, source_batch_invariant=source_batch_invariant)
        mlp = DeepseekV3MoERunner(trace, model, pool, prefix + ":mlp", group, collective,
            layer=model.num_hidden_layers, resolution=resolution, invocation=invocation_id)
        self.mtp_block = DeepseekV3Decoder(trace, model, pool, prefix + ":decoder", attention=attention, mlp=mlp)
        self.residual_add = SymbolicAdd(trace, prefix + ":residual-add", pool)
        self.head_norm = SymbolicRMSNorm(trace, prefix + ":head-norm", pool, model.hidden_size, eps)
        self.head = SymbolicTPLogitsProcessor(trace, prefix + ":head", pool,
            hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather,
            model_execution_version="vllm_0_27_1")

    def __call__(self, input_ids, positions, previous_hidden_states, *, position_values):
        rows, model = self.rows, self.model
        for value in (input_ids, positions, previous_hidden_states):
            self.trace._check_tensor(value)
        if (input_ids.shape != (rows,) or input_ids.dtype != "int32"
                or positions.shape != (rows,) or positions.dtype != "int64"
                or previous_hidden_states.shape != (rows, model.hidden_size) or previous_hidden_states.dtype != "bf16"
                or not isinstance(position_values, tuple)
                or any(type(p) is not int for p in position_values)
                or position_values != self.position_values):
            raise ValueError("V3 draft body requires its own typed IDs, positions, previous hidden and exact contexts")
        embedded = self.embed_tokens(input_ids)
        embedded = self.position_mask(positions, embedded, position_values=position_values,
            model_execution_version="vllm_0_27_1")
        embedded, previous = self.enorm(embedded), self.hnorm(previous_hidden_states)
        hidden = self.eh_proj(self.concat(embedded, previous))[0]
        hidden, residual = self.mtp_block(positions, hidden, None)
        pre_final_hidden = self.residual_add(residual, hidden)
        return pre_final_hidden, self.head_norm(pre_final_hidden)

    def compute_logits(self, pre_final_hidden):
        return self.head(self.head_norm(pre_final_hidden))


def trace_deepseek_v3_decode(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
        cache_capacity_tokens, source_profile, target_sampling_source=None,
        source_max_model_len=None, max_batched_tokens=None):
    """Official V3 prefill/mixed/decode source with optional explicit greedy.

    Inputs are already in decode-prefix source order (query threshold one),
    without an invented persistent InputBatch or runtime reorder receipt.
    """
    config = pool.parallel_plan.config
    source011 = source_profile == SOURCE_PROFILE_011
    source_version = "vllm_0_11_0" if source011 else "vllm_0_27_1"
    sampler = "vllm_0_11_plain_greedy_bf16" if source011 else "vllm_0_27_1_plain_greedy_bf16"
    if source011 and config.tensor_parallel_size != 1:
        raise NotImplementedError("native0.11 MLA source currently requires TP1")
    if (source_profile not in (SOURCE_PROFILE, SOURCE_PROFILE_011) or model.model_id != "deepseek-ai/DeepSeek-V3"
            or model.model_type != "deepseek_v3" or model.num_hidden_layers != 61
            or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
            or pool.hardware_id != "nvidia_h200" or config.tensor_parallel_size not in (1, 8)
            or len(pool.ranks) != config.tensor_parallel_size
            or config.enable_expert_parallel or any(value != 1 for value in (config.pipeline_parallel_size,
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("DeepSeek-V3 requires official FP8 H200 PP1 EP-off with "
            "explicit0.11.0 TP1 or0.27.1 TP1/8 source; "
            f"received source_profile={source_profile!r}")
    if (step.pool_name != pool.name or not step.requests or any(
            not ((r.phase == "decode" and r.scheduled_tokens == 1 and r.computed_tokens > 0)
                or (r.phase == "prefill" and r.scheduled_tokens > 0 and r.computed_tokens >= 0))
            or r.mtp_proposed_tokens or r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests)):
        raise NotImplementedError("DeepSeek-V3 requires ordinary single-token decode and/or prefill with MTP off")
    if target_sampling_source not in ((sampler,) if source011 else (None, sampler)):
        raise NotImplementedError(
            f"DeepSeek-V3 {'0.11.0' if source011 else '0.27.1'} target sampling requires "
            f"plain greedy source {sampler!r}; received {target_sampling_source!r}")
    rows = sum(r.scheduled_tokens for r in step.requests)
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(value) is not int for value in values):
            raise ValueError(f"DeepSeek-V3 {name} requires explicit scheduled metadata")
    if (any(not 0 <= value < model.vocab_size for value in input_ids)
            or positions != tuple(p for r in step.requests
                for p in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens))
            or any(not 0 <= value < model.max_position_embeddings for value in positions)):
        raise ValueError("DeepSeek-V3 IDs/positions must match its checkpoint and source contexts")
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    num_decodes = next((index for index, q in enumerate(q_lens) if q > 1), len(q_lens))
    if any(q == 1 for q in q_lens[num_decodes:]):
        raise NotImplementedError("DeepSeek-V3 requires source decode-prefix order: query-length one before prefill")
    if source_max_model_len is not None and (type(source_max_model_len) is not int
            or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings):
        raise ValueError("DeepSeek-V3 source maximum length must contain every logical context")
    if max_batched_tokens is not None and (type(max_batched_tokens) is not int or max_batched_tokens < rows):
        raise ValueError("DeepSeek-V3 source batch limit must contain all scheduled rows")
    contexts = tuple(r for r in step.requests if r.scheduled_tokens > 1 and r.computed_tokens > 0)
    if contexts:
        if source_max_model_len is None:
            raise NotImplementedError("DeepSeek-V3 context prefill requires an explicit source context limit")
        if sum(request.computed_tokens for request in contexts) > min(8 * source_max_model_len, 65536):
            raise NotImplementedError("DeepSeek-V3 context exceeds the source guaranteed single-chunk workspace")
        starts, cursor = [], 0
        for request in step.requests:
            starts.append(slot_mapping[cursor] - request.computed_tokens)
            cursor += request.scheduled_tokens
        if (any(start < 0 for start in starts)
                or any(start + k > following for start, k, following in zip(starts, kv_lens, starts[1:]))
                or slot_mapping != tuple(start + position for start, q, k in zip(starts, q_lens, kv_lens)
                    for position in range(k - q, k))):
            raise ValueError("DeepSeek-V3 context requires disjoint logical slots, not native page IDs")
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slot_mapping)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slot_mapping)):
        raise ValueError("DeepSeek-V3 cache requires declared logical contexts and distinct update slots")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = None if len(traces) == 1 else ForwardCollectiveGroup(traces, namespace=step.step_id)
    collective = None if group is None else pool.collective_for("all_reduce")
    gather = None if group is None else pool.collective_for("all_gather")
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)

    def forward(trace):
        network = DeepseekV3Model(trace, model, pool.name, group, collective, input_ids=input_ids,
            kv_lens=kv_lens, slots=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
            resolutions=resolutions, invocations=invocations, q_lens=q_lens, num_decodes=num_decodes,
            context_chunk_rows=None if source_max_model_len is None else min(8 * source_max_model_len, 65536),
            source_version=source_version)
        hidden = network(trace.tensor("input_ids", (rows,), "int32"), trace.tensor("positions", (rows,), "int64"))
        indices = trace.tensor("logits_indices", (len(q_lens),), "int32")
        ends, offset = [], 0
        for length in q_lens:
            offset += length
            ends.append(offset - 1)
        selected = SymbolicLogitsRowSelect(trace, "logits-row-select", pool.name)(hidden, indices,
            selected_indices=tuple(ends), model_execution_version=source_version, hardware_id=pool.hardware_id)
        logits = (SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
            model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected)[0] if group is None else
            SymbolicTPLogitsProcessor(trace, "lm-head", pool.name, hidden_size=model.hidden_size,
                vocab_size=model.vocab_size, group=group, config=gather, model_execution_version=source_version)(selected))
        if target_sampling_source is not None:
            sampled = SymbolicGreedy(trace, "target-sampler", pool.name)(logits, model_execution_version=source_version)

    for trace in traces:
        forward(trace)
        for identifier, node in tuple(trace._nodes.items()):
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "symbolic_mla_fresh_prefill_mathematical", "symbolic_mla_context_prefill_mathematical")):
                trace._nodes[identifier] = replace(node, attributes={**node.attributes,
                    "source_model_profile": source_profile, "source_tensor_parallel_size": config.tensor_parallel_size})
    return traces[0] if group is None else group


def trace_deepseek_v3_mtp_forward(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
        cache_capacity_tokens, source_profile, source_max_model_len, max_batched_tokens, max_requests,
        draft_input_token_ids, block_size_tokens, attention_source, source_batch_invariant=None):
    """Complete synchronous0.27 version-bound padded greedy source, batched/K2.

    Fresh target sampling or verification/rejection, first all-row draft and recurrent draft
    execute on real virtual ports. Draft IDs/acceptance are explicit workload
    scenarios, never numerically inferred. Native/provider, asynchronous host
    service and normal allocator capacity are independent unsupported gates.
    """
    from .traced_sampling import MTPSourceOps, prepare_padded_drafter, sample_padded_raw_logits
    from src.serving.schedule.schedule_step import schedule_step_sha256
    config = pool.parallel_plan.config
    flashmla = attention_source == "vllm_0_27_1_causal_flashmla_bf16_source"
    if (source_profile != SOURCE_PROFILE or model.model_id != "deepseek-ai/DeepSeek-V3"
            or model.model_type != "deepseek_v3" or model.num_hidden_layers != 61
            or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
            or pool.hardware_id != "nvidia_h200" or config.tensor_parallel_size != 8
            or config.enable_expert_parallel or any(value != 1 for value in (config.pipeline_parallel_size,
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
            or attention_source not in ("vllm_0_27_1_causal_triton_mla_source", "vllm_0_27_1_causal_flashmla_bf16_source")):
        raise NotImplementedError("V3 MTP requires explicit0.27.1 causal MLA source, official FP8 H200 TP8 EP-off")
    if flashmla and source_batch_invariant is not False:
        raise NotImplementedError("V3 FlashMLA source requires explicit BATCH_INVARIANT=False")
    if not flashmla and source_batch_invariant is not None:
        raise ValueError("V3 BATCH_INVARIANT declaration requires the explicit FlashMLA source")
    if step.pool_name != pool.name or not step.requests:
        raise ValueError("V3 MTP source requires a nonempty matching request pool")
    batch = len(step.requests)
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    verified = tuple(r.mtp_verified_tokens for r in step.requests)
    accepted = tuple(r.mtp_accepted_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    rows = sum(q_lens)
    fresh = all(r.phase == "prefill" and r.computed_tokens == 0
        and r.mtp_verified_tokens == r.mtp_proposed_tokens == r.mtp_accepted_tokens == 0
        and r.raw_model_output_tokens in (0, 1) and r.scheduled_tokens > 1 for r in step.requests)
    if flashmla and (len(set(q_lens)) != 1 or max(q_lens) > 128):
        raise NotImplementedError("V3 FlashMLA source requires uniform queries<=128")
    if (any(r.mtp_proposer_mode != "padded" or r.mtp_proposer_tokens != 2 for r in step.requests)
            or (not fresh and any(r.phase != "decode" or v < 1 or q != v + 1 or not 0 <= ac <= v
                or r.raw_model_output_tokens != ac + 1 or r.computed_tokens <= 0
                for r, q, v, ac in zip(step.requests, q_lens, verified, accepted)))):
        raise NotImplementedError("V3 MTP source requires fresh prompts or full verify rows and two padded drafts")
    next_positions = tuple(k - r.mtp_rejected_tokens for r, k in zip(step.requests, kv_lens))
    if (any(type(cap) is not int or cap < 1 for cap in
            (source_max_model_len, max_batched_tokens, max_requests, block_size_tokens))
            or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
            or rows > max_batched_tokens or batch > max_requests or max(next_positions) >= source_max_model_len):
        raise ValueError("V3 MTP source capacities must contain every target and draft position")
    if not flashmla and sum(r.computed_tokens for r in step.requests) > min(8 * source_max_model_len, 65536):
        raise NotImplementedError("V3 MTP target context exceeds the guaranteed single-chunk source workspace")
    # Each logical request owns its entire target/recurrent arena. An all-
    # accepted request may append beyond its target extent; reserve that row
    # before placing the next request, rather than aliasing its context.
    arena_starts, offset = [], 0
    row_starts, cursor = [], 0
    for q, k, following in zip(q_lens, kv_lens, next_positions):
        arena_starts.append(offset)
        row_starts.append(cursor)
        offset += max(k, following + 1)
        cursor += q
    retained = tuple(start + q - r.mtp_rejected_tokens - 1
        for start, q, r in zip(row_starts, q_lens, step.requests))
    expected_positions = tuple(p for q, k in zip(q_lens, kv_lens) for p in range(k - q, k))
    expected_slots = tuple(start + p for start, q, k in zip(arena_starts, q_lens, kv_lens)
        for p in range(k - q, k))
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < offset
            or not isinstance(slot_mapping, tuple) or any(type(s) is not int for s in slot_mapping)
            or not isinstance(positions, tuple) or any(type(p) is not int for p in positions)
            or slot_mapping != expected_slots or positions != expected_positions):
        raise ValueError("V3 MTP requires complete disjoint logical cache positions, not native page IDs")
    if (not isinstance(input_ids, tuple) or len(input_ids) != rows
            or any(type(token) is not int or not 0 <= token < model.vocab_size for token in input_ids)
            or not isinstance(draft_input_token_ids, tuple) or len(draft_input_token_ids) != 2
            or any(not isinstance(ids, tuple) for ids in draft_input_token_ids)
            or tuple(len(ids) for ids in draft_input_token_ids) != (rows, batch)
            or any(type(token) is not int or not 0 <= token < model.vocab_size
                for ids in draft_input_token_ids for token in ids)):
        raise ValueError("V3 MTP requires explicit typed target IDs and independent per-pass draft IDs")
    if any(draft_input_token_ids[0][i] != input_ids[i + 1] for i in range(rows - 1) if i not in retained):
        raise ValueError("V3 first-pass draft IDs disagree with the source shift at non-sampled rows")
    target_values = tuple(start + i for start, count in zip(row_starts, verified) for i in range(count))
    bonus_values = tuple(start + count for start, count in zip(row_starts, verified))
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    collective, gather = pool.collective_for("all_reduce"), pool.collective_for("all_gather")
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
    draft_resolutions = tuple(expert_routing.resolve(step_id=step.step_id, layer_index=None,
        moe_invocation_id=f"{step.step_id}:mtp-pass-{index}:moe", schedule_step_sha256=schedule_step_sha256(step),
        tokens=count, num_experts=model.num_experts, topk=model.num_experts_per_tok,
        model_config_sha256=model.config_sha256, pool_name=pool.name, tensor_parallel_size=8,
        expert_parallel_size=1, expert_rank_group=(pool.ranks[0],), expert_placement_strategy="linear",
        moe_layer_ordinal=58, assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
            if expert_routing.source_kind == "explicit_assumption" else None)) for index, count in enumerate((rows, batch)))

    def forward(trace):
        ops = MTPSourceOps(trace, "mtp-source", pool.name, model_execution_version="vllm_0_27_1")
        ids = trace.tensor("input_ids", (rows,), "int32")
        pos = trace.tensor("positions", (rows,), "int64")
        # Fresh prompts have no SpecDecodeMetadata: only request-end logits
        # are sampled. Verify keeps the five metadata transfers and explicit
        # all-row gather of _calc_spec_decode_metadata.
        selected_rows = retained if fresh else tuple(range(rows))
        indices = ops.h2d(trace.tensor("target.logits-indices", (len(selected_rows),), "int32"))
        prefix = None
        if not fresh:
            prefix = ops.h2d(trace.tensor("target.cu-draft-tokens", (batch,), "int32"))
            sampled_prefix = ops.h2d(trace.tensor("target.cu-sampled-tokens", (batch,), "int32"))
            target_indices = ops.h2d(trace.tensor("target.target-indices", (sum(verified),), "int32"))
            bonus_indices = ops.h2d(trace.tensor("target.bonus-indices", (batch,), "int32"))
            selected_ids = ops.gather(ids, indices, selected_indices=selected_rows)
            shifted_indices = ops.unary(target_indices, expression="add", scalar=1)
            previous_draft = ops.gather(selected_ids, shifted_indices, selected_indices=tuple(i + 1 for i in target_values))
        network = DeepseekV3Model(trace, model, pool.name, group, collective, input_ids=input_ids,
            kv_lens=kv_lens, slots=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
            resolutions=resolutions, invocations=invocations, q_lens=q_lens, num_decodes=batch if flashmla else 0,
            attention_source=attention_source, source_batch_invariant=source_batch_invariant,
            context_chunk_rows=min(8 * source_max_model_len, 65536))
        hidden = network(ids, pos)
        selected_hidden = ops.gather(hidden, indices, selected_indices=selected_rows)
        logits = SymbolicTPLogitsProcessor(trace, "lm-head", pool.name, hidden_size=model.hidden_size,
            vocab_size=model.vocab_size, group=group, config=gather, model_execution_version="vllm_0_27_1")(selected_hidden)
        if fresh:
            sampled = SymbolicGreedy(trace, "target-sampler", pool.name)(logits,
                model_execution_version="vllm_0_27_1")
        else:
            sampled = sample_padded_raw_logits(ops, logits, prefix=prefix, previous_draft=previous_draft,
                target_indices=target_indices, bonus_indices=bonus_indices, verified=verified, accepted=accepted,
                target_values=target_values, bonus_values=bonus_values)
        backup = ops.h2d(trace.tensor("mtp.backup-next-token", (max_requests,), "int32", persistent=True).narrow(0, 0, batch))
        discard = trace.tensor("parent.discard-mask", (max_requests,), "bool", persistent=True).narrow(0, 0, batch)
        query_start = trace.tensor("parent.query-start", (max_requests + 1,), "int32", persistent=True).narrow(0, 0, batch + 1)
        next_ids, sample_indices, rejected = prepare_padded_drafter(ops, sampled, discard, backup, query_start,
            vocab_size=model.vocab_size, valid_sampled_counts=tuple(ac + 1 for ac in accepted), discarded_rows=(False,) * batch, prefix=prefix)
        #0.27 padded mode passes views of all target tensors; unlike0.11 it
        # does not materialize all-row advanced-index copies here.
        draft_ids = trace.tensor("mtp.input-ids", (max_batched_tokens,), "int32", persistent=True)
        draft_pos = trace.tensor("mtp.positions", (max_batched_tokens,), "int64", persistent=True)
        draft_hidden = trace.tensor("mtp.hidden-states", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
        unused_embeds = trace.tensor("mtp.inputs-embeds", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
        arange = trace.tensor("mtp.arange", (max(max_requests + 1, max_batched_tokens),), "int32", persistent=True)
        first_ids, first_pos, first_hidden = draft_ids.narrow(0, 0, rows), draft_pos.narrow(0, 0, rows), draft_hidden.narrow(0, 0, rows)
        if rows > 1:
            ops.copy(ids.narrow(0, 1, rows - 1), first_ids.narrow(0, 0, rows - 1))
        ops.scatter(next_ids, sample_indices, first_ids, selected_indices=retained)
        ops.copy(pos, first_pos)
        ops.copy(hidden, first_hidden)
        predictor = DeepseekV3MTPPredictorLayer(trace, model, pool.name, "mtp-pass-0", group=group,
            collective=collective, gather=gather, input_token_ids=draft_input_token_ids[0], q_lens=q_lens,
            kv_lens=kv_lens, slots=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
            resolution=draft_resolutions[0], invocation_id=f"{step.step_id}:mtp-pass-0:moe",
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            attention_source=attention_source, source_batch_invariant=source_batch_invariant)
        pre_final, recycle = predictor(first_ids, first_pos, first_hidden, position_values=positions)
        sample_hidden = ops.gather(pre_final, sample_indices, selected_indices=retained)
        selected_positions = ops.gather(first_pos, sample_indices, selected_indices=retained)
        recurrent_hidden = ops.gather(recycle, sample_indices, selected_indices=retained)
        first_draft = ops.unary(predictor.compute_logits(sample_hidden), "argmax")
        seq_lens = trace.tensor("parent.seq-lens", (max_requests,), "int32", persistent=True).narrow(0, 0, batch)
        if rejected is not None:
            ops.binary(seq_lens, rejected, "subtract", inplace=True)
        blocks = trace.tensor("parent.block-table", (max_requests, -(-source_max_model_len // block_size_tokens)), "int32", persistent=True).narrow(0, 0, batch)
        slots_buffer = trace.tensor("mtp.slot-mapping-buffer", (max_batched_tokens,), "int64", persistent=True)
        out_pos, out_slots = draft_pos.narrow(0, 0, batch), slots_buffer.narrow(0, 0, batch)
        next_draft_id = ops.unary(first_draft, "cast_ids")
        ops.step_metadata(selected_positions, blocks, seq_lens, out_positions=out_pos, out_slots=out_slots,
            block_size=block_size_tokens, max_model_len=source_max_model_len)
        ops.copy(next_draft_id, draft_ids.narrow(0, 0, batch))
        ops.copy(recurrent_hidden, draft_hidden.narrow(0, 0, batch))
        recurrent = DeepseekV3MTPPredictorLayer(trace, model, pool.name, "mtp-pass-1", group=group,
            collective=collective, gather=gather, input_token_ids=draft_input_token_ids[1], q_lens=(1,) * batch,
            kv_lens=tuple(p + 1 for p in next_positions),
            slots=tuple(start + p for start, p in zip(arena_starts, next_positions)), cache_capacity_tokens=cache_capacity_tokens,
            resolution=draft_resolutions[1], invocation_id=f"{step.step_id}:mtp-pass-1:moe",
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            attention_source=attention_source, source_batch_invariant=source_batch_invariant,
            cache=predictor.cache, rotary_cache=predictor.rotary_cache,
            slot_tensor=out_slots)
        pre_final_next, recycle_next = recurrent(draft_ids.narrow(0, 0, batch), out_pos,
            draft_hidden.narrow(0, 0, batch), position_values=next_positions)
        second_draft = ops.unary(recurrent.compute_logits(pre_final_next), "argmax")
        output = ops.stack((first_draft, second_draft))

    for trace in traces:
        forward(trace)
    return group


def build_deepseek_v3_mtp_workload(step, model, deployment, *, launch_policy, runtime_inputs,
        kv_cache_events=(), trace_observer=None):
    """Production boundary for the explicit0.27 padded source projection."""
    from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
    if launch_policy != "analytical_launch" or deployment.mode != "colocated":
        raise NotImplementedError("V3 MTP source requires colocated analytical launch projection")
    if trace_observer is not None and not callable(trace_observer):
        raise TypeError("symbolic trace observer must be callable")
    if any(not isinstance(event, KVCacheEvent) for event in kv_cache_events):
        raise TypeError("V3 symbolic workload requires normalized cache events")
    if any(event.action in (KVCacheAction.LOAD, KVCacheAction.EVICT) for event in kv_cache_events):
        raise NotImplementedError("V3 MTP host KV transfers are not source-bound")
    if not step.requests:
        raise ValueError("V3 MTP source requires a nonempty request batch")
    pool = deployment.worker_pools[step.pool_name]
    if pool.kv_cache is None:
        raise ValueError("V3 MTP requires its explicit HBM KV block-size configuration")
    block_size = next(tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm")
    from .runtime_inputs import V32PaddedMTPInputs
    frame = runtime_inputs.model_source_inputs
    if frame is not None:
        if (not isinstance(frame, V32PaddedMTPInputs)
                or frame.request_ids != tuple(request.request_id for request in step.requests)
                or frame.block_size_tokens != block_size):
            raise ValueError("V3 current padded frame must match request order and configured cache block size")
        draft_ids = frame.draft_input_token_ids
    else:
        draft_ids = runtime_inputs.v3_mtp_draft_input_token_ids
    positions, slots, capacity = [], [], 0
    for request in step.requests:
        total = request.computed_tokens + request.scheduled_tokens
        next_position = total - request.mtp_rejected_tokens
        active = tuple(range(request.computed_tokens, total))
        positions.extend(active)
        slots.extend(capacity + position for position in active)
        capacity += max(total, next_position + 1)
    source = trace_deepseek_v3_mtp_forward(model, step, pool, runtime_inputs.expert_routing,
        input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
        cache_capacity_tokens=capacity, source_profile=runtime_inputs.symbolic_forward_source,
        source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
        max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
        max_requests=runtime_inputs.symbolic_source_max_requests,
        draft_input_token_ids=draft_ids,
        block_size_tokens=block_size, attention_source=runtime_inputs.v3_mtp_attention_source,
        source_batch_invariant=runtime_inputs.v3_mtp_batch_invariant)
    graph = source.graph
    if trace_observer is not None:
        trace_observer(source, graph)
    return graph
