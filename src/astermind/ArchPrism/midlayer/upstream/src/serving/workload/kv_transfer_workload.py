"""Request-scoped PD cache transport, independent of model forward construction."""

from src.serving.deployment.deployment_config import worker_role_phase
from src.serving.deployment.deployment_plan import DeploymentPlan, pd_kv_transfer_rank_pairs
from .model_config import ModelConfig
from .workload_graph import WorkloadGraph, WorkloadKind, WorkloadNode


def build_pd_kv_transfer_workload(
    request_id: str,
    context_tokens: int,
    model: ModelConfig,
    deployment: DeploymentPlan,
    source_pool_name: str,
    destination_pool_name: str,
) -> WorkloadGraph:
    """Build one request-scoped P-to-D KV transfer after routing."""

    if not isinstance(request_id, str) or not request_id:
        raise ValueError("KV transfer request_id must be non-empty")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
        raise TypeError("KV transfer context_tokens must be an integer")
    if context_tokens <= 0:
        raise ValueError("KV transfer context_tokens must be positive")
    if deployment.mode not in ("pd_disagg", "pd_af_disagg"):
        raise ValueError("KV transfer workload requires a PD deployment")
    if source_pool_name not in deployment.worker_pools:
        raise ValueError(f"unknown KV source pool {source_pool_name!r}")
    if destination_pool_name not in deployment.worker_pools:
        raise ValueError(
            f"unknown KV destination pool {destination_pool_name!r}")
    source = deployment.worker_pools[source_pool_name]
    destination = deployment.worker_pools[destination_pool_name]
    if worker_role_phase(source.role) != "prefill":
        raise ValueError("KV transfer source must have a prefill role")
    if worker_role_phase(destination.role) != "decode":
        raise ValueError("KV transfer destination must have a decode role")
    if (model.model_type in ("glm_moe_dsa", "deepseek_v3")
            and source.parallel_plan.config.tensor_parallel_size
                != destination.parallel_plan.config.tensor_parallel_size):
        raise NotImplementedError(
            "heterogeneous PD tensor parallel KV transfer has no explicit "
            "payload contract; residency is not a transfer-size substitute")
    if model.attention_layout == "dsa_mla":
        source_parallel = source.parallel_plan.config
        destination_parallel = destination.parallel_plan.config
        # The DSA transfer payload is EP-independent: the compressed MLA
        # ``kv_c||k_pe`` vector and the FP8 index cache are both
        # num_kv_heads=1 caches replicated on every TP rank and transferred
        # REPLICATE-style per TP rank pair, while expert parallelism only
        # shards MoE expert weights and never enters the KV/index-cache path
        # (pinned vLLM GLM-5.2/DeepSeek-V3.2 semantics: MLAAttentionSpec
        # num_kv_heads=1, indexer cache 132 B/token per full-indexer layer,
        # NIXL "cache is duplicated" rank mapping).  EP8 and EP1 therefore
        # share the identical per-rank payload and identity rank mapping.
        tp8_ep8 = (
            source_parallel.enable_expert_parallel
            and destination_parallel.enable_expert_parallel
            and source_parallel.expert_parallel_size == 8
            and destination_parallel.expert_parallel_size == 8
        )
        tp8_ep1 = (
            not source_parallel.enable_expert_parallel
            and not destination_parallel.enable_expert_parallel
            and source_parallel.expert_parallel_size == 1
            and destination_parallel.expert_parallel_size == 1
        )
        supported_pd_scope = (
            source_parallel.tensor_parallel_size == 8
            and destination_parallel.tensor_parallel_size == 8
            and (tp8_ep8 or tp8_ep1)
        )
        if not supported_pd_scope:
            raise NotImplementedError(
                "DSA PD transfer requires a combined compressed MLA and "
                "FP8 index-cache transfer contract")
    elif model.model_type == "deepseek_v3":
        source_parallel = source.parallel_plan.config
        destination_parallel = destination.parallel_plan.config
        # DeepSeek-V3 plain-MLA PD transfer payload: the compressed MLA
        # ``kv_c||k_pe`` vector is a num_kv_heads=1 cache replicated on
        # every TP rank and transferred REPLICATE-style per TP rank pair,
        # exactly like the DSA MLA subset above but without the GLM-5.2
        # FP8 index cache (DeepSeek-V3 has no DSA indexer).  Expert
        # parallelism only shards MoE expert weights and never enters the
        # KV path (pinned vLLM DeepSeek-V3 MLAAttentionSpec num_kv_heads=1,
        # NIXL "cache is duplicated" rank mapping); the validated
        # DeepSeek-V3 scope is the H200 InferenceX TP8/PP1/EP1 topology.
        supported_pd_scope = (
            source_parallel.tensor_parallel_size == 8
            and destination_parallel.tensor_parallel_size == 8
            and not source_parallel.enable_expert_parallel
            and not destination_parallel.enable_expert_parallel
            and source_parallel.expert_parallel_size == 1
            and destination_parallel.expert_parallel_size == 1
        )
        if not supported_pd_scope:
            raise NotImplementedError(
                "DeepSeek-V3 PD transfer requires a compressed MLA "
                "kv_c||k_pe transfer contract over TP8/EP1 pools")
    rank_pairs = pd_kv_transfer_rank_pairs(
        source, destination, model.num_key_value_heads)
    routes = [
        route for route in deployment.routes
        if route.payload == "kv"
        and route.source_pool == source_pool_name
        and route.destination_pool == destination_pool_name
    ]
    if len(routes) != 1:
        raise ValueError(
            "request-scoped KV transfer requires exactly one route from "
            f"{source_pool_name!r} to {destination_pool_name!r}")
    route = routes[0]
    source_tensor_parallel_size = (
        source.parallel_plan.config.tensor_parallel_size)
    destination_tensor_parallel_size = (
        destination.parallel_plan.config.tensor_parallel_size)
    base_node_id = (
        f"{request_id}:pd-kv-transfer:"
        f"{source_pool_name}-to-{destination_pool_name}"
    )
    nodes = []
    for edge_index, rank_pair in enumerate(rank_pairs):
        source_rank = rank_pair.source_rank
        destination_rank = rank_pair.destination_rank
        overlapping_kv_heads = (
            rank_pair.kv_head_end - rank_pair.kv_head_start)
        if model.attention_layout == "standard_gqa":
            bytes_per_token = (
                2 * model.num_hidden_layers * overlapping_kv_heads
                * model.head_dim * model.dtype_bytes)
        else:
            bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(
                source_tensor_parallel_size,
                source.parallel_plan.config.pipeline_parallel_size)
        node_id = (
            base_node_id
            if len(rank_pairs) == 1
            else f"{base_node_id}:kv-edge-{edge_index}"
        )
        nodes.append(WorkloadNode(
            node_id=node_id,
            kind=WorkloadKind.PD_KV_TRANSFER,
            pool_name=source_pool_name,
            shape={},
            rank=source_rank,
            attributes={
                "algorithm": route.algorithm,
                "bytes": context_tokens * bytes_per_token,
                "bytes_semantics": "payload",
                "backend": route.backend,
                "backend_config_sha256": route.backend_config_sha256,
                "direction": "point_to_point",
                "link": route.link,
                "topology": route.topology,
                "participants": (source_rank, destination_rank),
                "operation": "send_recv",
                "source_rank": source_rank,
                "destination_rank": destination_rank,
                "ring_order": (),
                "reduction": None,
                "resource_id": route.resource_id,
                "destination_pool": destination_pool_name,
                "rank_mapping": "kv_head_overlap",
                "source_logical_rank": rank_pair.source_logical_rank,
                "destination_logical_rank": (
                    rank_pair.destination_logical_rank),
                "source_tensor_parallel_size": source_tensor_parallel_size,
                "destination_tensor_parallel_size": (
                    destination_tensor_parallel_size),
                "kv_head_start": rank_pair.kv_head_start,
                "kv_head_end": rank_pair.kv_head_end,
                "num_key_value_heads": model.num_key_value_heads,
            },
        ))
    return WorkloadGraph(tuple(nodes))
