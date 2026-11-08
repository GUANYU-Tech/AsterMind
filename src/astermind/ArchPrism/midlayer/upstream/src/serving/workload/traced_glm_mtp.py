"""V32 eager startup and legacy padded MTP on shared virtual forwards.

Adapt v0.30 gpu_model_runner, llm_base_proposer and deepseek_v32/nvidia/mtp.
Input IDs and acceptance are explicit scenarios. Logical page coordinates,
source metadata math and complete useful operator work do not establish a
native allocator, provider, launch or observed runtime execution.
"""

from src.serving.schedule.schedule_step import schedule_step_sha256

from .runtime_inputs import ExpertRoutingInput, V32PaddedMTPInputs, vllm_0_30_source_model_inputs
from .traced_forward import ForwardCollectiveGroup, ForwardTrace
from .traced_models import DeepseekV32Model, DeepseekV32MTPPredictorLayer, _glm_source_weight_dtype, _source_expert_resolutions
from .traced_operators import SymbolicGreedy, SymbolicLinear, SymbolicTPLogitsProcessor
from .traced_sampling import MTPSourceOps, _mtp_retained_last_rows, prepare_padded_drafter, sample_padded_raw_logits
from .workload_graph import WorkloadKind


class GLMStartupAttentionFactory:
    """Explicit eager/no-metadata storage program, never request attention."""

    def __init__(self, *, tokens, source_max_model_len, indexer_max_logits_mb):
        if any(type(value) is not int or value <= 0 for value in
                (tokens, source_max_model_len, indexer_max_logits_mb)):
            raise ValueError("GLM startup requires explicit positive constructor limits and indexer logits MiB")
        self.tokens, self.max_model_len, self.logits_mb = tokens, source_max_model_len, indexer_max_logits_mb

    def validate(self, *, rows, max_batched_tokens, source_max_model_len,
                 cache_capacity_tokens, slot_mapping, selection_source, prefill_attention_source):
        if (rows != self.tokens or max_batched_tokens != self.tokens
                or source_max_model_len != self.max_model_len
                or any(value is not None for value in (cache_capacity_tokens, slot_mapping,
                    selection_source, prefill_attention_source))):
            raise ValueError("GLM startup factory requires the full constructor batch and no request KV/slot/attention metadata")

    def attention(self, trace, model, pool, prefix, *, layer_index, tp_size,
                  workspace, rotary_cache, topk_buffer):
        return _GLMStartupAttention(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=tp_size, workspace=workspace, rotary_cache=rotary_cache,
            topk_buffer=topk_buffer, factory=self)


class _GLMStartupAttention:
    """attention.forward prefix plus sparse indexer profile reservation.

    Same source linear operators and Q absorption as requests. The no-metadata
    branch passes no cache or slots to fused_norm_rope, reserves the indexer
    workspace, zeroes its existing output, and returns through o_proj.
    These storage-only nodes carry no timing/provider claim.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size,
                 workspace, rotary_cache, topk_buffer, factory):
        from .traced_models import _source_linear
        from .traced_operators import SymbolicBatchedLinear, SymbolicDSAQuery
        from .traced_sampling import StorageOnlyOps
        self.trace, self.model, self.prefix = trace, model, prefix
        self.heads, self.factory, self.workspace = model.num_attention_heads // tp_size, factory, workspace
        self.topk_buffer, self.rope = topk_buffer, rotary_cache
        self.has_indexer = layer_index == model.num_hidden_layers or model.indexer_types[layer_index] == "full"
        self.ops = StorageOnlyOps(trace, prefix + ":startup", pool)
        path = f"model.layers.{layer_index}.self_attn"
        self.qkv = _source_linear(trace, model, pool, prefix + ":qkv-a", model.hidden_size,
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim, path + ".fused_qkv_a_proj")
        self.q_b = _source_linear(trace, model, pool, prefix + ":q-b", model.q_lora_rank,
            self.heads * (model.qk_nope_head_dim + model.qk_rope_head_dim), path + ".q_b_proj")
        self.q_absorb = SymbolicBatchedLinear(trace, prefix + ":q-absorb", pool,
            self.heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.o_proj = _source_linear(trace, model, pool, prefix + ":o-proj", self.heads * model.v_head_dim,
            model.hidden_size, path + ".o_proj")
        self.wk, self.index_q = None, None
        if self.has_indexer:
            self.wk = _source_linear(trace, model, pool, prefix + ":wk-weights", model.hidden_size,
                model.index_head_dim + model.index_n_heads, path + ".indexer.wk_weights_proj")
            self.index_q = _source_linear(trace, model, pool, prefix + ":index-q", model.q_lora_rank,
                model.index_n_heads * model.index_head_dim, path + ".indexer.wq_b")
        dummy_q = trace.tensor(trace.unique_name("startup.dummy-index-q"), (1, 1, 1), "bf16", persistent=True)
        dummy_weights = trace.tensor(trace.unique_name("startup.dummy-index-weights"), (1, 1), "fp32", persistent=True)
        self.fused_q = SymbolicDSAQuery(trace, prefix + ":fused-q", pool, model,
            has_indexer=self.has_indexer, rotary_cache=rotary_cache, index_rotary_cache=rotary_cache,
            dummy_index_q=dummy_q, dummy_index_weights=dummy_weights)

    def _reserve_indexer(self):
        # indexer.get_max_prefill_buffer_size returns 40*runtime max len.
        # No divisor for GLM's uncompressed indexer. UINT8 dummy logits are
        # local to sparse_attn_indexer, while the workspace manager retains
        # its arena after the temporary views have been discarded.
        capacity = 40 * self.factory.max_model_len
        self.workspace.get_simultaneous(((capacity, self.model.index_head_dim), "fp8"),
            ((capacity, 4), "uint8"), ((1024 * 1024,), "uint8"))
        dummy_logits = self.trace.tensor(self.trace.unique_name(self.prefix + ":profile-logits"),
            (self.factory.logits_mb * 1024 * 1024,), "uint8")

    def __call__(self, positions, hidden_states):
        trace, model, rows = self.trace, self.model, self.factory.tokens
        if hidden_states.shape != (rows, model.hidden_size) or positions.shape != (rows,):
            raise ValueError("GLM no-metadata startup attention requires its full constructor batch")
        qkv = self.qkv(hidden_states)[0]
        q_c, kv_c, k_pe = qkv.split((model.q_lora_rank, model.kv_lora_rank, model.qk_rope_head_dim))
        kw, index_k, index_weights = None, None, None
        if self.has_indexer:
            kw = self.wk(hidden_states)[0]
            index_k, index_weights = kw.split((model.index_head_dim, model.index_n_heads))
        output = trace.tensor(trace.unique_name(self.prefix + ":output"), (rows, self.heads * model.v_head_dim), "bf16")
        kv_out = trace.tensor(trace.unique_name(self.prefix + ":kv-out"), kv_c.shape, "bf16")
        rope_out = trace.tensor(trace.unique_name(self.prefix + ":rope-out"), k_pe.shape, "bf16")
        inputs = (positions, q_c, kv_c, k_pe, self.rope)
        if index_k is not None:
            inputs += (index_k,)
        q_c = self.ops.operation("no_metadata_fused_norm_rope", inputs, shape=q_c.shape,
            dtype="bf16", writes=(kv_out, rope_out, self.topk_buffer) if self.has_indexer else (kv_out, rope_out))
        q = self.q_b(q_c)[0].view_as_shape(rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
        q_nope, q_pe = q.split((model.qk_nope_head_dim, model.qk_rope_head_dim))
        ql_nope = self.q_absorb(q_nope.transpose(0, 1)).transpose(0, 1)
        index_q = None if self.index_q is None else self.index_q(q_c)[0].view_as_shape(
            rows, model.index_n_heads, model.index_head_dim)
        index_fp8, weights_out, mqa_q = self.fused_q(positions, q_pe, ql_nope, index_q, index_weights)
        if self.has_indexer:
            self._reserve_indexer()
        self.ops.operation("no_metadata_attention_zero_output", (), shape=output.shape, dtype=output.dtype, out=output)
        return self.o_proj(output)[0]


def trace_glm_startup_profile(model, pool, expert_routing, *, tokens, max_requests,
        source_max_model_len, workspace_initial_capacity_bytes, indexer_max_logits_mb,
        source_profile, include_mtp, num_speculative_tokens):
    """Full target, optional sequential NextN dummy forwards, then sampler.

    v0.30 eager CP/PP1 EP-off startup only. Draft forwards all consume the
    constructor's full token buffers; they neither compact to request count
    nor sample/reject tokens, and do not recycle the previous pass's result.
    Routing remains an explicit conditional input, not a proven global bound.
    """
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
    from .traced_sampling import VLLM030WorkerStorage, worker_profile_storage_forward
    config = pool.parallel_plan.config
    weight_dtype = _glm_source_weight_dtype(source_profile)
    if (model.model_type != "glm_moe_dsa" or model.num_hidden_layers != 78
            or model.compute_dtype != "bf16" or model.weight_dtype != weight_dtype
            or pool.hardware_id != "nvidia_h200" or config.tensor_parallel_size not in (1, 8)
            or len(pool.ranks) != config.tensor_parallel_size or config.enable_expert_parallel
            or any(value != 1 for value in (config.pipeline_parallel_size, config.sequence_parallel_size,
                config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("GLM startup requires accurate BF16-compute H200 TP1/8 PP/SP/CP/DP1 EP-off source")
    if (type(include_mtp) is not bool or type(num_speculative_tokens) is not int
            or (include_mtp and num_speculative_tokens != 2) or (not include_mtp and num_speculative_tokens != 0)):
        raise NotImplementedError("GLM startup declares target-off or two sequential eager MTP passes explicitly")
    factory = GLMStartupAttentionFactory(tokens=tokens, source_max_model_len=source_max_model_len,
        indexer_max_logits_mb=indexer_max_logits_mb)
    if (type(max_requests) is not int or max_requests <= 0 or source_max_model_len > model.max_position_embeddings
            or not isinstance(expert_routing, ExpertRoutingInput)
            or not isinstance(workspace_initial_capacity_bytes, tuple)
            or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("GLM startup requires explicit request/context limits, routing and per-rank workspace history")
    batch = min(tokens, max_requests)
    lengths = (tokens // batch,) * (batch - 1) + (tokens // batch + tokens % batch,)
    if max(lengths) > source_max_model_len:
        raise ValueError("GLM startup context must cover every dummy query")
    step = ScheduleStep("glm-startup-profile", pool.name, tuple(ScheduledRequest(str(i), "prefill", q, 0)
        for i, q in enumerate(lengths)))
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, tokens)
    draft_resolution = None
    if include_mtp:
        draft_resolution = expert_routing.resolve(step_id=step.step_id, layer_index=None,
            moe_invocation_id=f"{step.step_id}:mtp-pass-0:moe", schedule_step_sha256=schedule_step_sha256(step),
            tokens=tokens, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name=pool.name, tensor_parallel_size=len(pool.ranks),
            expert_parallel_size=1, expert_rank_group=(pool.ranks[0],), expert_placement_strategy="linear",
            moe_layer_ordinal=sum(model.layer_uses_moe(layer) for layer in range(model.num_hidden_layers)),
            assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
                if expert_routing.source_kind == "explicit_assumption" else None))
    traces = tuple(ForwardTrace(rank=rank, namespace=f"glm-startup:rank-{rank}") for rank in pool.ranks)
    group = None if len(traces) == 1 else ForwardCollectiveGroup(traces, namespace=step.step_id)
    collective = None if group is None else pool.collective_for("all_reduce")
    gather = None if group is None else pool.collective_for("all_gather")

    def rank_profile(trace, capacity):
        network = DeepseekV32Model(trace, model, pool.name, group, collective, token_ids=(0,) * tokens,
            q_lens=lengths, kv_lens=lengths, request_phases=("prefill",) * batch, slot_mapping=None,
            cache_capacity_tokens=None, source_max_model_len=source_max_model_len, max_batched_tokens=tokens,
            resolutions=resolutions, invocation_ids=invocations, workspace_initial_capacity_bytes=capacity,
            selection_source=None, source_profile=source_profile, prefill_attention_source=None,
            model_input_order=None, startup_attention_factory=factory)
        worker = VLLM030WorkerStorage(trace, tokens=tokens, max_requests=max_requests,
            inputs_embeds_size=model.hidden_size, dtype=model.compute_dtype)
        head = (SymbolicLinear(trace, "profile.lm-head", pool.name, model.hidden_size, model.vocab_size,
            kind=WorkloadKind.LM_HEAD) if group is None else SymbolicTPLogitsProcessor(trace, "profile.lm-head", pool.name,
                hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather))
        draft_forward = None
        if include_mtp:
            draft_ids = trace.tensor("profile.mtp.input_ids", (tokens,), "int32", persistent=True)
            draft_pos = trace.tensor("profile.mtp.positions", (tokens,), "int64", persistent=True)
            draft_hidden = trace.tensor("profile.mtp.hidden_states", (tokens, model.hidden_size), "bf16", persistent=True)
            draft_embeds = trace.tensor("profile.mtp.inputs_embeds", (tokens, model.hidden_size), "bf16", persistent=True)
            draft_arange = trace.tensor("profile.mtp.arange", (max(max_requests + 1, tokens),), "int32", persistent=True)
            draft_backup = trace.tensor("profile.mtp.backup_next_token_ids", (max_requests,), "int32", persistent=True)
            draft_slots = trace.tensor("profile.mtp.slot_mapping", (tokens,), "int64", persistent=True)
            draft_topk = trace.tensor("profile.mtp.topk", (tokens, model.index_topk), "int32", persistent=True)
            predictor = DeepseekV32MTPPredictorLayer(trace, model, pool.name, "profile.mtp", group=group,
                collective=collective, gather=gather, input_token_ids=(0,) * tokens, q_lens=lengths, kv_lens=lengths,
                request_phases=("prefill",) * batch, slots=None, cache_capacity_tokens=None,
                resolution=draft_resolution, invocation_id=f"{step.step_id}:mtp-pass-0:moe", source_profile=source_profile,
                source_max_model_len=source_max_model_len, max_batched_tokens=tokens, workspace=network.workspace,
                topk_buffer=draft_topk, cache=None, index_cache=None, rotary_cache=network.rotary_cache,
                index_rotary_cache=network.index_rotary_cache, slot_tensor=None, indexer_type="full",
                prefill_attention_source=None, selection_source=None, startup_attention_factory=factory)

            def draft_forward():
                trace.checkpoint("worker.drafter.begin")
                for index in range(num_speculative_tokens):
                    trace.checkpoint(f"worker.drafter.pass-{index}.begin")
                    # The return value is discarded before the next call.
                    predictor(draft_ids, draft_pos, draft_hidden, position_values=(0,) * tokens)
                    trace.checkpoint(f"worker.drafter.pass-{index}.end")
                trace.checkpoint("worker.drafter.end")

        compute_logits = (lambda x: head(x)[0]) if group is None else head
        trace.checkpoint("worker.profile.begin")
        worker_profile_storage_forward(trace, network, compute_logits, input_ids=worker.input_ids,
            positions=worker.positions, query_lengths=lengths,
            sampler_profile="vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise",
            pool_name=pool.name, drafter_dummy_run=draft_forward)
        trace.checkpoint("worker.profile.end")

    for trace, capacity in zip(traces, workspace_initial_capacity_bytes, strict=True):
        rank_profile(trace, capacity)
    return traces[0] if group is None else group


def trace_glm_dsa_mtp_forward(model, step, pool, expert_routing, *, input_ids, source_profile,
                            source_max_model_len, max_batched_tokens, max_requests,
                            workspace_initial_capacity_bytes, selection_source,
                            prefill_attention_source, source_inputs):
    """Full target → greedy rejection → declared sequential padded drafts.

    Both target and drafter retain speculative_config and its decode threshold.
    Draft IDs explicitly declare the pass count, including the source's
    one-draft early return. No numerical draft or acceptance is generated.
    A mixed step may have fresh or preemption-replay requests whose proposals
    are not consumed. The source still forwards their complete first-pass
    rows, then one row per incoming request on each remaining pass. It does
    not compact the physical drafter batch to the proposal consumers.
    """
    config = pool.parallel_plan.config
    if (model.model_type != "glm_moe_dsa" or model.mtp_architecture != "glm_moe_dsa_mtp"
            or model.num_hidden_layers != 78 or model.num_mtp_layers != 1
            or model.compute_dtype != "bf16" or model.weight_dtype != _glm_source_weight_dtype(source_profile)
            or pool.hardware_id != "nvidia_h200" or config.tensor_parallel_size not in (1, 8)
            or len(pool.ranks) != config.tensor_parallel_size or config.enable_expert_parallel
            or any(value != 1 for value in (config.pipeline_parallel_size, config.sequence_parallel_size,
                config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("V32 padded source requires its H200 BF16-compute TP1/TP8 PP/SP/CP/DP1 EP-off checkpoint")
    if not isinstance(source_inputs, V32PaddedMTPInputs) or not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("V32 padded source requires typed independent draft and routing scenarios")
    if source_inputs.request_ids != tuple(request.request_id for request in step.requests) or step.pool_name != pool.name:
        raise ValueError("V32 padded scenario must identify the complete incoming request order")
    if (any(type(value) is not int or value < 1 for value in (source_max_model_len, max_batched_tokens, max_requests))
            or source_max_model_len > model.max_position_embeddings or len(step.requests) > max_requests
            or sum(request.scheduled_tokens for request in step.requests) > max_batched_tokens):
        raise ValueError("V32 padded source requires explicit constructor limits covering all rows and requests")
    if (not isinstance(workspace_initial_capacity_bytes, tuple)
            or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("V32 padded source requires explicit per-rank workspace entry history")
    drafts = source_inputs.draft_input_token_ids
    num_drafts = len(drafts)
    consumers = tuple(request for request in step.requests if request.mtp_proposer_tokens)
    disabled = tuple(request for request in step.requests if request.mtp_proposer_mode == "disabled")
    # gpu_model_runner.propose_draft_token_ids keeps the complete input
    # batch in prepare_inputs_padded; llm_base_proposer.set_inputs_first_pass
    # globally shifts all target rows and replaces one retained row per
    # request. K=1 returns only after that full forward and full-batch sample;
    # K>1 keeps the same batch size for rollback and every one-row recurrence.
    # Disabled rows are therefore an output-consumer distinction, not
    # permission to drop their physical work. A non-discarded sampled
    # candidate is distinct from a consumed output/proposal: raw output 0
    # stays 0 and no candidate IDs are exported as consumed for that request.
    # The independent typed scenario supplies IDs, never a generated value.
    if any(not ((request.phase == "prefill" and request.computed_tokens == 0)
                or (request.phase == "decode" and request.resumed_from_preemption
                    and request.recomputed_tokens == request.scheduled_tokens))
            or request.raw_model_output_tokens not in (0, 1)
            or any((request.mtp_proposed_tokens, request.mtp_verified_tokens, request.mtp_accepted_tokens,
                request.mtp_proposer_tokens, request.mtp_next_proposed_tokens))
            for request in disabled):
        raise NotImplementedError("V32 mixed disabled requests require fresh or explicit full preemption-replay rows")
    if (not isinstance(input_ids, tuple) or any(type(token) is not int or not 0 <= token < model.vocab_size for token in input_ids)
            or not consumers
            or any(request.mtp_proposer_mode != "padded" or request.mtp_proposer_tokens != num_drafts
                or request.raw_model_output_tokens != request.mtp_accepted_tokens + 1
                for request in consumers)
            or len(consumers) + len(disabled) != len(step.requests)
            or any(request.scheduled_tokens < request.mtp_verified_tokens + 1 for request in step.requests)):
        raise NotImplementedError("V32 source requires complete padded draft rows matching its declared passes, valid anchors and target IDs")
    # This synchronous source has one pending sampled token before the
    # scheduled draft suffix (scheduler.py:656-660,813-823). Extra target
    # rows need an explicit recomputation domain; they are not additional
    # speculative slots with no declared draft token.
    if any(request.phase == "decode" and request.recomputed_tokens == 0
            and request.scheduled_tokens != request.mtp_verified_tokens + 1 for request in consumers):
        raise ValueError("MTP target verification must schedule one terminal token after the verified draft tokens")
    block_size = source_inputs.block_size_tokens
    if pool.kv_cache is not None:
        actual_block_size = next(tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm")
        if block_size != actual_block_size:
            raise ValueError("V32 logical source block size disagrees with its pool cache")
    step, input_ids, ordering = vllm_0_30_source_model_inputs(step, embedding_token_ids=input_ids,
        prompt_lengths=None, num_speculative_tokens=num_drafts)
    requests = step.requests
    q_lens = tuple(request.scheduled_tokens for request in requests)
    kv_lens = tuple(request.computed_tokens + request.scheduled_tokens for request in requests)
    phases = tuple(ordering["model_attention_phases"])
    rows, batch = sum(q_lens), len(q_lens)
    verified = tuple(request.mtp_verified_tokens for request in requests)
    accepted = tuple(request.mtp_accepted_tokens for request in requests)
    retained = _mtp_retained_last_rows(requests)
    positions = tuple(position for request in requests for position in range(request.computed_tokens,
        request.computed_tokens + request.scheduled_tokens))
    last_positions = tuple(positions[index] + num_drafts - 1 for index in retained)
    if max((*kv_lens, *(position + 1 for position in last_positions))) > source_max_model_len:
        raise NotImplementedError("V32 cap-exceeded padded positions are outside the complete source domain")
    if (tuple(len(ids) for ids in drafts) != (rows, *((batch,) * (num_drafts - 1)))
            or any(token >= model.vocab_size for ids in drafts for token in ids)):
        raise ValueError("V32 scenario must supply independent in-vocabulary IDs for every row of every draft pass")
    if any(drafts[0][index] != input_ids[index + 1] for index in range(rows - 1) if index not in retained):
        raise ValueError("V32 first-pass draft IDs disagree with the source global shift at non-sampled rows")
    # Disjoint logical per-request page arenas; no native block IDs or
    # KVAllocation are invented. A final recurrent write may extend a page.
    bases, arena_size = [], 0
    for length in kv_lens:
        bases.append(arena_size)
        arena_size += -(-(length + num_drafts - 1) // block_size) * block_size
    slots = tuple(bases[index] + request.computed_tokens + offset for index, request in enumerate(requests)
        for offset in range(request.scheduled_tokens))
    old_selected_contexts = tuple(positions[index] + 1 for index in retained)
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = None if len(traces) == 1 else ForwardCollectiveGroup(traces, namespace=step.step_id)
    collective = None if group is None else pool.collective_for("all_reduce")
    gather = None if group is None else pool.collective_for("all_gather")
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
    draft_resolutions = tuple(expert_routing.resolve(step_id=step.step_id, layer_index=None,
        moe_invocation_id=f"{step.step_id}:mtp-pass-{index}:moe", schedule_step_sha256=schedule_step_sha256(step),
        tokens=count, num_experts=model.num_experts, topk=model.num_experts_per_tok,
        model_config_sha256=model.config_sha256, pool_name=pool.name, tensor_parallel_size=len(pool.ranks),
        expert_parallel_size=1, expert_rank_group=(pool.ranks[0],), expert_placement_strategy="linear",
        moe_layer_ordinal=sum(model.layer_uses_moe(layer) for layer in range(model.num_hidden_layers)),
        assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
            if expert_routing.source_kind == "explicit_assumption" else None))
        for index, count in enumerate((rows, *((batch,) * (num_drafts - 1)))))

    def rank_forward(trace, capacity):
        ops = MTPSourceOps(trace, "mtp-source", pool.name, model_execution_version="vllm_0_30_0")
        ids = trace.tensor("input_ids", (rows,), "int32")
        pos = trace.tensor("positions", (rows,), "int64")
        selected_rows, target_values, bonus_values = [], [], []
        offset, logits_offset = 0, 0
        for query, count in zip(q_lens, verified, strict=True):
            selected_rows.extend(range(offset + query - count - 1, offset + query))
            target_values.extend(range(logits_offset, logits_offset + count))
            bonus_values.append(logits_offset + count)
            offset += query
            logits_offset += count + 1
        indices = ops.h2d(trace.tensor("target.logits-indices", (len(selected_rows),), "int32"))
        prefix = None
        if any(verified):
            prefix = ops.h2d(trace.tensor("target.cu-draft-tokens", (batch,), "int32"))
            sampled_prefix = ops.h2d(trace.tensor("target.cu-sampled-tokens", (batch,), "int32"))
            target_indices = ops.h2d(trace.tensor("target.target-indices", (sum(verified),), "int32"))
            bonus_indices = ops.h2d(trace.tensor("target.bonus-indices", (batch,), "int32"))
            selected_ids = ops.gather(ids, indices, selected_indices=tuple(selected_rows))
            shifted = ops.unary(target_indices, expression="add", scalar=1)
            previous_draft = ops.gather(selected_ids, shifted,
                selected_indices=tuple(index + 1 for index in target_values))
        network = DeepseekV32Model(trace, model, pool.name, group, collective, token_ids=input_ids,
            q_lens=q_lens, kv_lens=kv_lens, request_phases=phases, slot_mapping=slots,
            cache_capacity_tokens=arena_size, source_max_model_len=source_max_model_len,
            max_batched_tokens=max_batched_tokens, resolutions=resolutions, invocation_ids=invocations,
            workspace_initial_capacity_bytes=capacity, selection_source=selection_source,
            source_profile=source_profile, prefill_attention_source=prefill_attention_source,
            model_input_order=ordering, num_speculative_tokens=num_drafts)
        hidden = network(ids, pos)
        selected_hidden = ops.gather(hidden, indices, selected_indices=tuple(selected_rows))
        head = (SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size, model.vocab_size,
            kind=WorkloadKind.LM_HEAD) if group is None else SymbolicTPLogitsProcessor(trace, "lm-head", pool.name,
                hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather))
        logits = head(selected_hidden)[0] if group is None else head(selected_hidden)
        if prefix is None:
            sampled = SymbolicGreedy(trace, "target-sampler", pool.name)(logits)
        else:
            sampled = sample_padded_raw_logits(ops, logits, prefix=prefix, previous_draft=previous_draft,
                target_indices=target_indices, bonus_indices=bonus_indices, verified=verified, accepted=accepted,
                target_values=tuple(target_values), bonus_values=tuple(bonus_values))
        backup = ops.h2d(trace.tensor("mtp.backup-next-token", (max_requests,), "int32", persistent=True).narrow(0, 0, batch))
        discard = trace.tensor("parent.discard-mask", (max_requests,), "bool", persistent=True).narrow(0, 0, batch)
        query_start = trace.tensor("parent.query-start", (max_requests + 1,), "int32", persistent=True).narrow(0, 0, batch + 1)
        next_ids, sample_indices, rejected = prepare_padded_drafter(ops, sampled, discard, backup, query_start,
            vocab_size=model.vocab_size, valid_sampled_counts=tuple(count + 1 for count in accepted),
            discarded_rows=(False,) * batch, prefix=prefix)
        draft_ids = trace.tensor("mtp.input-ids", (max_batched_tokens,), "int32", persistent=True)
        draft_pos = trace.tensor("mtp.positions", (max_batched_tokens,), "int64", persistent=True)
        draft_hidden = trace.tensor("mtp.hidden-states", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
        unused_embeds = trace.tensor("mtp.inputs-embeds", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
        arange = trace.tensor("mtp.arange", (max(max_requests + 1, max_batched_tokens),), "int32", persistent=True)
        first_ids = draft_ids.narrow(0, 0, rows)
        first_pos = draft_pos.narrow(0, 0, rows)
        first_hidden = draft_hidden.narrow(0, 0, rows)
        if rows > 1:
            ops.copy(ids.narrow(0, 1, rows - 1), first_ids.narrow(0, 0, rows - 1))
        ops.scatter(next_ids, sample_indices, first_ids, selected_indices=retained)
        ops.copy(pos, first_pos)
        ops.copy(hidden, first_hidden)
        mtp_cache = trace.tensor("mtp.kv-cache", (arena_size, model.kv_lora_rank + model.qk_rope_head_dim), "bf16", persistent=True)
        mtp_index_cache = trace.tensor("mtp.index-cache", (arena_size, 132), "uint8", persistent=True)
        topk = trace.tensor("mtp.topk", (max_batched_tokens, model.index_topk), "int32", persistent=True)
        arenas = dict(cache_capacity_tokens=arena_size, source_profile=source_profile,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            workspace=network.workspace, topk_buffer=topk, cache=mtp_cache, index_cache=mtp_index_cache,
            rotary_cache=network.rotary_cache, index_rotary_cache=network.index_rotary_cache,
            selection_source=selection_source, num_speculative_tokens=num_drafts,
            group=group, collective=collective, gather=gather)
        predictor = DeepseekV32MTPPredictorLayer(trace, model, pool.name, "mtp-pass-0",
            input_token_ids=drafts[0], q_lens=q_lens, kv_lens=kv_lens, request_phases=phases,
            slots=slots, slot_tensor=network.slot_mapping, indexer_type="full",
            resolution=draft_resolutions[0], invocation_id=f"{step.step_id}:mtp-pass-0:moe",
            prefill_attention_source=prefill_attention_source, **arenas)
        post_norm, recycle = predictor(first_ids, first_pos, first_hidden, position_values=positions)
        # llm_base_proposer only toggles skip_topk and compacts the first
        # pass's selection when the draft checkpoint enables index sharing.
        if model.index_share_for_mtp_iteration:
            compacted = ops.gather(topk, sample_indices, selected_indices=retained)
            ops.copy(compacted, topk.narrow(0, 0, batch))
        sample_hidden = ops.gather(post_norm, sample_indices, selected_indices=retained)
        if num_drafts == 1:
            # llm_base_proposer.py returns before position/hidden gathers,
            # seq-len rollback, metadata updates, casts or stack allocation.
            first_draft = ops.unary(predictor.compute_logits(sample_hidden), "argmax")
            output = first_draft.view_as_shape(batch, 1)
            return
        selected_positions = ops.gather(first_pos, sample_indices, selected_indices=retained)
        recurrent_hidden = ops.gather(recycle, sample_indices, selected_indices=retained)
        first_draft = ops.unary(predictor.compute_logits(sample_hidden), "argmax")
        seq_lens = trace.tensor("parent.seq-lens", (max_requests,), "int32", persistent=True).narrow(0, 0, batch)
        if rejected is not None:
            ops.binary(seq_lens, rejected, "subtract", inplace=True)
        blocks = trace.tensor("parent.block-table", (max_requests, -(-source_max_model_len // block_size)),
            "int32", persistent=True).narrow(0, 0, batch)
        slots_buffer = trace.tensor("mtp.slot-mapping-buffer", (max_batched_tokens,), "int64", persistent=True)
        out_pos, out_slots = draft_pos.narrow(0, 0, batch), slots_buffer.narrow(0, 0, batch)
        draft_outputs = [first_draft]
        for draft_index in range(1, num_drafts):
            next_id_values = ops.unary(draft_outputs[-1], "cast_ids")
            ops.step_metadata(selected_positions, blocks, seq_lens, out_positions=out_pos, out_slots=out_slots,
                block_size=block_size, max_model_len=source_max_model_len)
            ops.copy(next_id_values, draft_ids.narrow(0, 0, batch))
            ops.copy(recurrent_hidden, draft_hidden.narrow(0, 0, batch))
            current_positions = tuple(positions[index] + draft_index for index in retained)
            current_slots = tuple(base + position for base, position in zip(bases, current_positions, strict=True))
            recurrent = DeepseekV32MTPPredictorLayer(trace, model, pool.name, f"mtp-pass-{draft_index}",
                input_token_ids=drafts[draft_index], q_lens=(1,) * batch,
                kv_lens=tuple(position + 1 for position in current_positions),
                request_phases=("decode",) * batch, slots=current_slots, slot_tensor=out_slots,
                indexer_type="shared" if model.index_share_for_mtp_iteration else "full",
                resolution=draft_resolutions[draft_index], invocation_id=f"{step.step_id}:mtp-pass-{draft_index}:moe",
                prefill_attention_source=None, input_norm=predictor.input_norm,
                reused_selection_kv_lens=old_selected_contexts if model.index_share_for_mtp_iteration else None, **arenas)
            final_hidden, recurrent_hidden = recurrent(draft_ids.narrow(0, 0, batch), out_pos,
                draft_hidden.narrow(0, 0, batch), position_values=current_positions)
            draft_outputs.append(ops.unary(recurrent.compute_logits(final_hidden), "argmax"))
            selected_positions = out_pos
        output = ops.stack(tuple(draft_outputs))

    for trace, capacity in zip(traces, workspace_initial_capacity_bytes, strict=True):
        rank_forward(trace, capacity)
    return traces[0] if group is None else group
