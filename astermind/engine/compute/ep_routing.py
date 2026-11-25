"""Build runtime WideEP transfers from the expert routing used by the source plan."""

from collections import Counter, defaultdict
from itertools import islice
import re

from src.serving.workload.runtime_inputs import _iter_seeded_expert_routes, _shared_routing_group_id
from src.serving.workload.workload_graph import WorkloadKind


_TRANSFER = re.compile(
    r"^(?P<prefix>.*[:\-])(?P<role>dispatch-[a-z]+|combine)(?P<ordinal>:[0-9]+)?$"
)
_DISPATCH_PAYLOADS = (
    frozenset(("dispatch-hidden", "dispatch-weights", "dispatch-ids")),
    frozenset(("dispatch-input", "dispatch-weights", "dispatch-ids")),
    frozenset(("dispatch-input", "dispatch-weights", "dispatch-ids", "dispatch-scales")),
    frozenset(("dispatch-latent", "dispatch-weights", "dispatch-ids")),
    frozenset(("dispatch-hidden", "dispatch-logits")),
)


def direct_ep_matrices(calls, execution, pool_name, step_id, *, replica_tokens=None,
                       source_nodes=None):
    """Return byte matrices for stage-local WideEP dispatches and combines."""
    pool = execution.deployment.worker_pools[pool_name]
    parallel = pool.parallel_plan.config
    if not parallel.enable_expert_parallel or parallel.data_parallel_size <= 1:
        return {}

    expert_calls, transfers = defaultdict(dict), defaultdict(dict)
    for call in calls.values():
        if call.pool_name != pool_name:
            continue
        source = call.memory_source
        if source is None and source_nodes is not None and len(call.source_node_ids) == 1:
            source = source_nodes.get(call.source_node_ids[0])
        if source is None:
            continue
        transfer = call.kernel_spec.transfer
        if source.kind is WorkloadKind.FUSED_MOE:
            attrs = source.attributes
            invocation = attrs["moe_invocation_id"]
            prior = expert_calls[invocation].setdefault(call.rank, source)
            if prior.node_id != source.node_id:
                raise ValueError("Direct WideEP found multiple experts for the same invocation and rank")
        elif (transfer is not None and transfer.operation in ("all_gatherv", "reduce_scatterv")
                and source.attributes.get("collective_group") in ("expert", "data")
                and len(transfer.participants) == parallel.expert_parallel_size):
            match = _TRANSFER.fullmatch(source.node_id)
            if match and (match["role"] != "combine" or "moe_invocation_id" in source.attributes):
                group = (match["prefix"], match["ordinal"])
                if match["role"] in transfers[group]:
                    raise ValueError("Direct WideEP found duplicate transfer roles")
                transfers[group][match["role"]] = (call, source.attributes)

    result, route_cache, completed_invocations = {}, {}, set()
    model = execution.model
    model_identity = model.expert_routing_assumption_semantics_sha256 or model.config_sha256
    for roles in transfers.values():
        if "combine" not in roles:
            raise ValueError("Direct WideEP dispatch has no matching combine")
        if frozenset(roles.keys() - {"combine"}) not in _DISPATCH_PAYLOADS:
            raise ValueError("Direct WideEP has an incomplete dispatch payload")
        combine, combine_attrs = roles["combine"]
        invocation = combine_attrs["moe_invocation_id"]
        ranks = combine.kernel_spec.transfer.participants
        by_rank = expert_calls[invocation]
        if by_rank.keys() != set(ranks) or invocation in completed_invocations:
            raise ValueError("Direct WideEP requires every expert owner and one combine per invocation")
        completed_invocations.add(invocation)

        source = by_rank[ranks[0]].attributes
        provenance = source["routing_provenance"]
        expected = source["tokens_per_expert"]
        owners = _expert_owners(by_rank, ranks, expected, provenance["record_sha256"])
        layer = provenance["layer_index"]
        if provenance["source_kind"] != "explicit_assumption" or provenance["step_id"] != step_id:
            raise NotImplementedError("Direct WideEP requires the current step's seeded per-token routing")
        stage = (pool.parallel_plan.layer_to_pipeline_stage[layer] if layer is not None
                 else pool.parallel_plan.config.pipeline_parallel_size - 1)
        seed_ranks = pool.physical_ranks_for_pipeline_stage(stage)
        if set(seed_ranks) != set(ranks):
            raise ValueError("Direct WideEP transfer disagrees with the source expert group")

        rows = combine_attrs["symbolic_variable_rows"]
        valid = rows if replica_tokens is None else _request_rows(pool, ranks, rows, replica_tokens)
        topk = model.num_experts_per_tok
        tokens, remainder = divmod(sum(expected), topk)
        if remainder or not tokens or tokens not in (sum(rows), sum(valid)):
            raise ValueError("Direct WideEP routing counts disagree with physical and valid token rows")
        routed_rows = rows if tokens == sum(rows) else valid

        routing_id = _shared_routing_group_id(
            step_id=step_id, layer_index=layer, tokens=tokens,
            num_experts=model.num_experts, topk=topk,
            model_config_sha256=model_identity, pool_name=pool_name,
            tensor_parallel_size=parallel.tensor_parallel_size,
            expert_parallel_size=parallel.expert_parallel_size, expert_rank_group=seed_ranks,
            expert_placement_strategy="linear",
        )
        key = (routing_id, owners, routed_rows, valid)
        if key not in route_cache:
            routes = _iter_seeded_expert_routes(
                tokens, model.num_experts, topk, source_id=provenance["source_id"],
                root_seed=provenance["root_seed"], moe_invocation_id=routing_id,
                distribution=provenance["distribution"], hot_expert_count=provenance["hot_expert_count"],
                hot_expert_weight=provenance["hot_expert_weight"],
            )
            matrix, counts = _token_matrix(routes, routed_rows, owners, valid)
            route_cache[key] = matrix, counts
        matrix, counts = route_cache[key]
        if counts != expected:
            raise ValueError("Direct WideEP token routes disagree with the predicted expert counts")

        for role, (call, _) in roles.items():
            transfer = call.kernel_spec.transfer
            returning = role == "combine"
            operation = "reduce_scatterv" if returning else "all_gatherv"
            row_bytes, extra = divmod(transfer.bytes, sum(rows))
            if (transfer.operation != operation or transfer.participants != ranks or extra or not row_bytes
                    or transfer.partition_bytes != tuple(count * row_bytes for count in rows)):
                raise ValueError(f"Unexpected Direct WideEP transfer contract: {call.call_id}")
            flow = zip(*matrix) if returning else matrix
            result[call.call_id] = tuple(tuple(count * row_bytes for count in row) for row in flow)
    if completed_invocations != expert_calls.keys() or not result:
        raise NotImplementedError("Direct WideEP requires a source plan with explicit dispatch and combine")
    return result


def _expert_owners(by_rank, ranks, expected, routing_record):
    owners = [-1] * len(expected)
    for index, rank in enumerate(ranks):
        attrs = by_rank[rank].attributes
        if (attrs["tokens_per_expert"] != expected
                or attrs["routing_provenance"]["record_sha256"] != routing_record):
            raise ValueError("Direct WideEP ranks disagree on the expert routing record")
        for expert in attrs["owned_experts"]:
            if not 0 <= expert < len(owners) or owners[expert] != -1:
                raise ValueError("Direct WideEP requires one owner per routed expert")
            owners[expert] = index
    if -1 in owners:
        raise ValueError("Direct WideEP expert placement is incomplete")
    return tuple(owners)


def _request_rows(pool, ranks, rows, replica_tokens):
    """Strip each replica's trailing TP padding and idle placeholder rows."""
    parallel = pool.parallel_plan.config
    replica_width = parallel.pipeline_parallel_size * parallel.tensor_parallel_size
    remaining = list(replica_tokens)
    ranked_rows = sorted(zip(ranks, rows, strict=True),
                         key=lambda item: pool.logical_rank_for_physical_rank(item[0]))
    valid_by_rank = {}
    for rank, count in ranked_rows:
        replica = pool.logical_rank_for_physical_rank(rank) // replica_width
        valid = min(count, remaining[replica])
        valid_by_rank[rank] = valid
        remaining[replica] -= valid
    if any(remaining):
        raise ValueError("Direct WideEP rank rows do not cover the scheduled requests")
    return tuple(valid_by_rank[rank] for rank in ranks)


def _token_matrix(routes, rows, owners, valid_rows):
    """Count one transfer per token and selected remote rank."""
    counts = Counter()
    matrix = [[0] * len(rows) for _ in rows]
    for source, (count, valid) in enumerate(zip(rows, valid_rows, strict=True)):
        for offset, selected in enumerate(islice(routes, count)):
            counts.update(selected)
            if offset < valid:
                for destination in {owners[expert] for expert in selected} - {source}:
                    matrix[source][destination] += 1
    return matrix, tuple(counts[expert] for expert in range(len(owners)))
