"""Adapt DeepSeek V4 WideEP transfers to the modeled token destinations."""

from collections import Counter, defaultdict
import re

from src.kernel.kernel_spec import KernelKind
from src.serving.workload.runtime_inputs import _shared_routing_group_id, seeded_expert_routes


_TRANSFER = re.compile(r":ep:layer-(\d+):ffn:(dispatch-hidden|dispatch-weights|dispatch-ids|combine)$")
_EXPERT = re.compile(r":layer-(\d+):ffn:routed$")
_ROLES = frozenset(("dispatch-hidden", "dispatch-weights", "dispatch-ids", "combine"))


def direct_ep_matrices(calls, execution, pool_name, step_id):
    """Return All-to-All byte matrices for the four communication calls per MoE layer."""
    model = execution.model
    if model.model_type != "deepseek_v4":
        return {}
    parallel = execution.deployment.worker_pools[pool_name].parallel_plan.config
    if not parallel.enable_expert_parallel or parallel.data_parallel_size <= 1:
        return {}
    if parallel.pipeline_parallel_size != 1 or parallel.tensor_parallel_size <= 1:
        raise NotImplementedError("Direct DeepSeek V4 EP requires multi-TP, multi-DP and PP1")
    routing = execution.runtime_inputs.expert_routing
    if routing.source_kind != "explicit_assumption":
        raise NotImplementedError("Direct DeepSeek V4 EP requires seeded token-level routes")

    transfers, expert_counts = defaultdict(dict), {}
    for call in calls.values():
        if call.pool_name != pool_name:
            continue
        spec = call.kernel_spec
        if spec.kind is KernelKind.COMMUNICATION and (match := _TRANSFER.search(call.call_id)):
            layer, role = int(match[1]), match[2]
            if role in transfers[layer]:
                raise ValueError(f"Duplicate DeepSeek V4 EP transfer for layer {layer}: {role}")
            transfers[layer][role] = call
        elif spec.kind is KernelKind.FUSED_MOE and (match := _EXPERT.search(call.call_id)):
            layer = int(match[1])
            counts = spec.attributes["tokens_per_expert"]
            if expert_counts.setdefault(layer, counts) != counts:
                raise ValueError(f"DeepSeek V4 EP ranks disagree on expert counts for layer {layer}")

    layers = set(range(model.num_hidden_layers))
    if set(transfers) != layers or set(expert_counts) != layers:
        raise ValueError("DeepSeek V4 plan must contain dispatch, expert compute and combine for every layer")

    cache, result = {}, {}
    for layer, roles in transfers.items():
        if roles.keys() != _ROLES:
            raise ValueError(f"Incomplete DeepSeek V4 EP transfer roles for layer {layer}")
        hidden = roles["dispatch-hidden"].kernel_spec.transfer
        ranks = hidden.participants
        if len(ranks) != parallel.expert_parallel_size or model.num_experts % len(ranks):
            raise NotImplementedError("Direct DeepSeek V4 EP requires equal linear expert placement")
        hidden_row_bytes = model.hidden_size * 2
        if len(hidden.partition_bytes) != len(ranks) or any(n % hidden_row_bytes for n in hidden.partition_bytes):
            raise ValueError("DeepSeek V4 EP hidden shards must contain whole BF16 token rows")
        rows = tuple(n // hidden_row_bytes for n in hidden.partition_bytes)
        tokens = sum(rows)
        expected = expert_counts[layer]
        if not tokens or tokens * model.num_experts_per_tok != sum(expected):
            raise NotImplementedError("Direct DeepSeek V4 EP requires unpadded token rows")

        group = _shared_routing_group_id(
            step_id=step_id, layer_index=layer, tokens=tokens,
            num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name=pool_name,
            tensor_parallel_size=parallel.tensor_parallel_size,
            expert_parallel_size=len(ranks), expert_rank_group=ranks,
            expert_placement_strategy="linear",
        )
        key = (group, rows)
        if key not in cache:
            routes = seeded_expert_routes(
                tokens, model.num_experts, model.num_experts_per_tok,
                source_id=routing.source_id, root_seed=routing.root_seed, moe_invocation_id=group,
                distribution=routing.distribution, hot_expert_count=routing.hot_expert_count,
                hot_expert_weight=routing.hot_expert_weight,
            )
            counts = Counter(expert for route in routes for expert in route)
            cache[key] = (
                tuple(counts[i] for i in range(model.num_experts)),
                _token_matrix(routes, rows, model.num_experts // len(ranks)),
            )
        counts, matrix = cache[key]
        if counts != expected:
            raise ValueError(f"Direct EP routing disagrees with predicted expert counts for layer {layer}")

        for role, call in roles.items():
            transfer = call.kernel_spec.transfer
            expected_operation = "reduce_scatterv" if role == "combine" else "all_gatherv"
            row_bytes, extra = divmod(transfer.bytes, tokens)
            if (transfer.operation != expected_operation or transfer.participants != ranks or extra
                    or transfer.partition_bytes != tuple(count * row_bytes for count in rows)):
                raise ValueError(f"Unexpected DeepSeek V4 EP transfer contract: {call.call_id}")
            flow = zip(*matrix) if role == "combine" else matrix
            result[call.call_id] = tuple(tuple(count * row_bytes for count in row) for row in flow)
    return result


def _token_matrix(routes, rows, experts_per_rank):
    """One token per destination rank, with local expert work staying local."""
    matrix = [[0] * len(rows) for _ in rows]
    offset = 0
    for source, count in enumerate(rows):
        for selected in routes[offset:offset + count]:
            destinations = {expert // experts_per_rank for expert in selected}
            for destination in destinations - {source}:
                matrix[source][destination] += 1
        offset += count
    return matrix
