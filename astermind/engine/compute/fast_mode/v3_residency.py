"""DeepSeek-V3 weight residency for Fast Mode's native Scheduler budget."""

from dataclasses import replace

from src.serving.kv_cache.model_residency import _linear_weight_bytes, model_rank_residencies
from src.serving.workload.model_config import (
    is_deepseek_v3_data_parallel_model_scope,
    is_deepseek_v3_parallel_model_scope,
)


def weight_bytes_by_rank(model, parallel, layers):
    tp = parallel.tensor_parallel_size
    dp = parallel.data_parallel_size
    ep = parallel.expert_parallel_size
    pp = parallel.pipeline_parallel_size
    if (parallel.sequence_parallel_size != 1 or parallel.context_parallel_size != 1
            or not (parallel.enable_expert_parallel and is_deepseek_v3_data_parallel_model_scope(model, parallel)
                    if dp > 1 else is_deepseek_v3_parallel_model_scope(
                        model, tp, pp, expert_parallel=parallel.enable_expert_parallel))):
        raise ValueError("unsupported DeepSeek-V3 Fast Mode residency layout")

    reference = (replace(parallel, data_parallel_size=1, enable_expert_parallel=False)
                 if dp > 1 else parallel)
    ranks = model_rank_residencies(
        model, reference, worker_role="colocated", pipeline_layer_partition=layers,
    )
    if dp == 1:
        return tuple(rank.weight_resident_bytes for rank in ranks)

    # Attention, dense layers, vocabulary and norms retain their TP/PP
    # layout. A WideEP rank owns E/(TP*DP) experts and a full shared expert.
    deltas = {rank.pipeline_parallel_rank: sum(
        _expert_delta(model, layer, tp, ep)
        for layer in range(rank.layer_start, rank.layer_end) if model.layer_uses_moe(layer)
    ) for rank in ranks if rank.tensor_parallel_rank == 0}
    return tuple(rank.weight_resident_bytes + deltas[rank.pipeline_parallel_rank]
                 for rank in ranks) * dp


def _expert_delta(model, layer, tp, ep):
    prefix = f"model.layers.{layer}.mlp"
    intermediate = model.moe_intermediate_size
    shared = intermediate * model.num_shared_experts

    def weights(path, width):
        return (_linear_weight_bytes(model, out_features=2 * width, in_features=model.hidden_size,
                                     module_path=f"{path}.gate_proj")
                + _linear_weight_bytes(model, out_features=model.hidden_size, in_features=width,
                                       module_path=f"{path}.down_proj"))

    experts = weights(f"{prefix}.experts.0", intermediate)
    return ((model.num_experts // ep - model.num_experts // tp) * experts
            + weights(f"{prefix}.shared_experts", shared)
            - weights(f"{prefix}.shared_experts", shared // tp))
