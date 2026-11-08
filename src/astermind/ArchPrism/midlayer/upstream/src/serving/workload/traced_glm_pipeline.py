"""GLM vLLM0.30 PP source projection using the shared virtual layers.

The upstream TP-partial hidden is transmitted without an added reduction.
Its default flattened TP lanes are gathered by the receiver, exactly as
the original transport requests. This does not prove numerical TP/PP
equivalence, provider selection, async retirement or normal capacity.
"""

from .traced_forward import ForwardCollectiveGroup, ForwardPipelineGroup, ForwardTrace
from .traced_models import DeepseekV32Model, _glm_source_weight_dtype, _plain_greedy_target, _run_vllm030_pipeline_stages
from .traced_operators import _dsa_source_phase


def trace_glm_pipeline_forward(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
        cache_capacity_tokens, source_max_model_len, max_batched_tokens, workspace_initial_capacity_bytes,
        selection_source, source_profile, target_sampling_source="vllm_0_30_plain_greedy_bf16",
        prefill_attention_source=None, source_prompt_lengths=None):
    """Execute existing TP8 PP source layouts, preserving the unclosed PP gate."""
    from .runtime_inputs import ExpertRoutingInput, vllm_0_30_source_model_inputs
    from src.serving.schedule.schedule_step import schedule_step_sha256
    config = pool.parallel_plan.config
    h200_pp2 = (pool.hardware_id == "nvidia_h200" and config.pipeline_parallel_size == 2
        and not config.enable_expert_parallel)
    a800_pp4 = (pool.hardware_id == "nvidia_a800" and config.pipeline_parallel_size == 4
        and config.enable_expert_parallel and config.expert_parallel_size == 8 and model.weight_dtype == "bf16")
    if (model.model_type != "glm_moe_dsa" or model.num_hidden_layers != 78
            or model.compute_dtype != "bf16" or model.weight_dtype != _glm_source_weight_dtype(source_profile)
            or not (h200_pp2 or a800_pp4) or config.tensor_parallel_size != 8 or not config.is_rank_local
            or any(value != 1 for value in (config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
            or len(pool.ranks) != 8 * config.pipeline_parallel_size or model.hidden_size % 8):
        raise NotImplementedError("GLM PP source projection requires existing H200 TP8/PP2 EP-off or A800 BF16 TP8/PP4/EP8, SP/CP/DP1")
    stage_ids = tuple(range(config.pipeline_parallel_size))
    assignments = pool.parallel_plan.layer_to_pipeline_stage
    if len(assignments) != model.num_hidden_layers or assignments != tuple(sorted(assignments)) or set(assignments) != set(stage_ids):
        raise ValueError("GLM pipeline requires the deployment's complete ordered layer assignments")
    ranges = tuple(tuple(i for i, owner in enumerate(assignments) if owner == stage) for stage in stage_ids)
    if any(model.indexer_types[indexes[0]] != "full" for indexes in ranges):
        raise NotImplementedError("each GLM stage must start with its source full indexer, rather than borrowing another stage's top-k")
    rows = sum(request.scheduled_tokens for request in step.requests)
    if (step.pool_name != pool.name or not step.requests or any(request.phase not in ("prefill", "decode")
            or request.scheduled_tokens <= 0 or request.mtp_proposer_tokens or request.mtp_proposed_tokens
            or request.mtp_verified_tokens for request in step.requests)):
        raise NotImplementedError("GLM pipeline projection requires ordinary MTP-off source requests")
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(value) is not int for value in values):
            raise ValueError(f"GLM pipeline {name} requires explicit integer metadata for every row")
    if (any(not 0 <= token < model.vocab_size for token in input_ids)
            or positions != tuple(p for request in step.requests for p in range(request.computed_tokens,
                request.computed_tokens + request.scheduled_tokens))):
        raise ValueError("GLM pipeline IDs/positions must match the checkpoint and scheduled contexts")
    step, input_ids, order = vllm_0_30_source_model_inputs(step, embedding_token_ids=input_ids, prompt_lengths=source_prompt_lengths)
    positions = tuple(positions[row] for row in order["token_row_permutation"])
    slot_mapping = tuple(slot_mapping[row] for row in order["token_row_permutation"])
    q_lens = tuple(request.scheduled_tokens for request in step.requests)
    kv_lens = tuple(request.computed_tokens + request.scheduled_tokens for request in step.requests)
    phases = tuple(order["model_attention_phases"])
    if prefill_attention_source is None and "prefill" in phases and all(q == k and k <= model.index_topk
            for q, k, phase in zip(q_lens, kv_lens, phases) if phase == "prefill"):
        prefill_attention_source = "dense_mha_fresh"
    _dsa_source_phase(q_lens, kv_lens, phases, prefill_attention_source)
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slot_mapping)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slot_mapping)
            or type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
            or type(max_batched_tokens) is not int or max_batched_tokens < rows):
        raise ValueError("GLM pipeline requires complete bounded cache/context/batch metadata")
    if (not isinstance(workspace_initial_capacity_bytes, tuple) or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("GLM pipeline requires explicit workspace entry capacities for every physical rank")
    if selection_source != "declared_causal_topk" or not isinstance(expert_routing, ExpertRoutingInput):
        raise ValueError("GLM pipeline requires explicit causal selection and canonical routing input")
    if target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16"):
        raise NotImplementedError("GLM pipeline only adapts the existing plain target sampler")
    stages = tuple(tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}")
        for rank in pool.physical_ranks_for_pipeline_stage(stage)) for stage in stage_ids)
    groups = tuple(ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:stage-{stage}")
        for stage, traces in enumerate(stages))
    pipeline = ForwardPipelineGroup(tuple(trace for traces in stages for trace in traces),
        namespace=step.step_id, collective_groups=groups)
    pipeline.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
        "pipeline_layer_partition": tuple(len(indexes) for indexes in ranges),
        "tp_partial_hidden_sent_without_added_reduction": True, "numerical_tp_pp_equivalence_bound": False,
        "runner_intermediate_copy_source": "vllm_0_30_gpu_model_runner_sync_and_gather_intermediate_tensors",
        "normal_capacity_accepted": False, "actual_provider_bound": False}
    reduce, gather = pool.collective_for("all_reduce"), pool.collective_for("all_gather")
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    digest = schedule_step_sha256(step)
    resolutions = tuple(expert_routing.resolve(step_id=step.step_id, layer_index=layer,
        moe_invocation_id=invocations[layer], schedule_step_sha256=digest, tokens=rows,
        num_experts=model.num_experts, topk=model.num_experts_per_tok, model_config_sha256=model.config_sha256,
        pool_name=pool.name, tensor_parallel_size=8, expert_parallel_size=config.expert_parallel_size,
        expert_rank_group=(groups[assignments[layer]].participants if config.enable_expert_parallel
            else (stages[assignments[layer]][0].rank,)), expert_placement_strategy="linear",
        moe_layer_ordinal=model.moe_layer_ordinal(layer), assumption_model_semantics_sha256=(
            model.expert_routing_assumption_semantics_sha256 if expert_routing.source_kind == "explicit_assumption" else None))
        if model.layer_uses_moe(layer) else None for layer in range(model.num_hidden_layers))

    def run_stage(trace, stage, incoming):
        network = DeepseekV32Model(trace, model, pool.name, groups[stage], reduce, token_ids=input_ids,
            q_lens=q_lens, kv_lens=kv_lens, request_phases=phases, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_max_model_len=source_max_model_len,
            max_batched_tokens=max_batched_tokens, resolutions=resolutions, invocation_ids=invocations,
            workspace_initial_capacity_bytes=workspace_initial_capacity_bytes[pool.ranks.index(trace.rank)],
            selection_source=selection_source, source_profile=source_profile,
            prefill_attention_source=prefill_attention_source, model_input_order=order,
            expert_parallel_size=config.expert_parallel_size,
            layer_range=(ranges[stage][0], ranges[stage][-1] + 1))
        pos = trace.tensor("positions", (rows,), "int64")
        ids = trace.tensor("input_ids", (rows,), "int32") if network.first_stage else None
        hidden = network(ids, pos, intermediate_tensors=incoming)
        if not network.last_stage:
            return hidden
        indices = trace.tensor("logits_indices", (len(q_lens),), "int32")
        _plain_greedy_target(trace, model, pool.name, groups[stage], gather, hidden, indices, q_lens,
            sample_logits=target_sampling_source == "vllm_0_30_plain_greedy_bf16",
            cast_logits=target_sampling_source == "vllm_0_30_plain_sampler_bf16")
        return None

    _run_vllm030_pipeline_stages(pipeline, pool, rows=rows, hidden_size=model.hidden_size,
        max_batched_tokens=max_batched_tokens, run_stage=run_stage)
    return pipeline
