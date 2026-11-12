"""O-S history/capacity contracts. No numerical KV tensors or replacement allocator."""
from dataclasses import asdict
import hashlib

from src.serving.kv_cache.kv_capacity import load_runtime_memory_policy
from src.serving.kv_cache.model_residency import model_rank_residencies
from src.serving.parallelism.parallel_config import ParallelConfig


class SessionHistoryStore:
    """Immutable, declared linear histories shared by independently resident Cells."""
    def __init__(self):
        self.contexts = {}
        self.tips = {}
        self.pending = {}

    def prepare(self, request):
        if request.session_id is None:
            if request.prefix_hit_tokens:
                raise ValueError("prefix reuse requires a declared session history")
            return None
        key = (request.session_id, request.context_id or request.request_id)
        if key in self.contexts or key in self.pending:
            raise ValueError("duplicate context identity")
        if any(k[0] == request.session_id for k in self.pending):
            raise ValueError("concurrent branches of one session are outside this phase")
        parent = request.parent_context_id
        if parent != self.tips.get(request.session_id):
            raise ValueError("parent must be the completed current session tip")
        previous = () if parent is None else self.contexts[(request.session_id, parent)]
        if request.input_tokens < len(previous):
            raise ValueError("linear continuation cannot truncate its parent history")
        keys = previous + tuple(f"{key[1]}:input:{i}" for i in range(len(previous), request.input_tokens))
        self.pending[key] = keys
        return {"session_id": key[0], "context_id": key[1], "token_keys": keys,
                "declared_prefix_hit_tokens": request.prefix_hit_tokens}

    def complete(self, request, generated):
        if request.session_id is None:
            return
        key = (request.session_id, request.context_id or request.request_id)
        keys = self.pending.pop(key)
        self.contexts[key] = keys + tuple(f"{key[1]}:output:{i}" for i in range(generated))
        self.tips[key[0]] = key[1]

    def abort(self, request):
        self.pending.pop((request.session_id, request.context_id or request.request_id), None)

    def history(self, session_id, context_id):
        return {"session_id": session_id, "context_id": context_id,
                "token_keys": self.contexts[(session_id, context_id)]}


def history_block_hashes(history, block_size, computed_length=None):
    """Content identities are symbolic declarations, never all-zero dummy IDs."""
    keys = history["token_keys"]
    limit = len(keys) if computed_length is None else min(len(keys), computed_length)
    parent = hashlib.sha256(history["session_id"].encode()).digest()
    result = []
    for start in range(0, limit - block_size + 1, block_size):
        digest = hashlib.sha256(parent)
        for key in keys[start:start + block_size]:
            encoded = key.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        parent = digest.digest()
        result.append(parent)
    return result


def capacity_inputs(config, model, hardware):
    parallelism = ParallelConfig(tensor_parallel_size=config.tensor_parallel_size,
                                 pipeline_parallel_size=config.pipeline_parallel_size,
                                 enable_expert_parallel=config.enable_expert_parallel)
    residencies = model_rank_residencies(model, parallelism, worker_role="colocated")
    rank_geometry = [dict(rank=r.rank, data_parallel_rank=(r.data_parallel_rank if hasattr(r, "data_parallel_rank") else 0),
                          layer_start=r.layer_start, layer_end=r.layer_end,
                          weight_resident_bytes=r.weight_resident_bytes,
                          kv_block_bytes=config.kv_block_size_tokens * model.kv_bytes_per_token_for_layer_range(
                              config.tensor_parallel_size, r.layer_start, r.layer_end)) for r in residencies]
    block_sizes = {r["kv_block_bytes"] for r in rank_geometry}
    if len(block_sizes) != 1:
        raise NotImplementedError("Cell capacity currently requires uniform per-rank KV geometry")
    block_bytes = rank_geometry[0]["kv_block_bytes"]
    if config.memory_policy_file is None:
        return {"source_mode": "debug_explicit_block_count", "num_kv_blocks": config.num_kv_blocks,
                "kv_block_bytes": block_bytes, "available_kv_bytes": config.num_kv_blocks * block_bytes,
                "rank_geometry": rank_geometry, "bytes_scope": "one_rank",
                "warning": "explicit test pool; not a hardware-derived HBM capacity"}
    policy_path = config.root / config.memory_policy_file
    policies = load_runtime_memory_policy(policy_path)
    policies.validate_scheduler_pools(("main",))
    policy = policies.policy_for_pool("main")
    binding = policy.binding
    hardware_hash = hashlib.sha256((config.root / config.hardware_file).read_bytes()).hexdigest()
    if binding.hardware_config_sha256 != hardware_hash:
        raise ValueError("memory policy hardware fingerprint differs from this Cell hardware")
    expected = (config.hardware_id, model.config_sha256, "colocated", "colocated",
                config.tensor_parallel_size, config.pipeline_parallel_size, parallelism.expert_parallel_size,
                config.max_model_len, config.max_batched_tokens, config.max_requests, config.kv_block_size_tokens,
                parallelism.data_parallel_size)
    actual = (binding.hardware_id, binding.model_config_sha256, binding.deployment, binding.worker_role,
              binding.tensor_parallel_size, binding.pipeline_parallel_size, binding.expert_parallel_size,
              binding.max_model_len, binding.max_batched_tokens, binding.max_requests, binding.kv_block_size_tokens,
              binding.data_parallel_size if hasattr(binding, "data_parallel_size") else 1)
    if expected != actual:
        raise ValueError("memory policy does not describe this Cell configuration")
    physical = hardware.resource_by_role("dram").size_bytes
    utilized = policy.utilized_hbm_bytes(physical)
    # The shared logical block pool must fit every rank, including the stage
    # that holds embedding/lm_head weights. Never sum HBM across GPUs.
    weights = max(r.weight_resident_bytes for r in residencies)
    available = utilized - weights - policy.non_kv_reserve_bytes
    if available <= 0:
        raise ValueError("HBM has no capacity remaining for KV after weights/reserves")
    return {"source_mode": "analytical_current_vllm_allocator", "physical_hbm_bytes": physical,
            "utilized_hbm_bytes": utilized, "weight_resident_bytes": weights,
            "non_kv_reserve_bytes": policy.non_kv_reserve_bytes, "available_kv_bytes": available,
            "kv_block_bytes": block_bytes, "policy_path": str(policy_path),
            "rank_geometry": rank_geometry, "bytes_scope": "one_rank_bottleneck",
            "policy_sha256": policies.source_sha256, "policy": asdict(policy)}
