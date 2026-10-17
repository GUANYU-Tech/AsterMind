"""Static, rank-local model-weight residency for KV capacity planning."""

from dataclasses import dataclass

from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.shape_inference import padded_vocab_size


@dataclass(frozen=True)
class ModelRankResidency:
    """Analytical model bytes resident on one physical inference rank."""

    rank: int
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
    expert_parallel_rank: int
    layer_start: int
    layer_end: int
    embedding_bytes: int
    decoder_layer_bytes: int
    mtp_bytes: int
    final_norm_bytes: int
    lm_head_bytes: int

    def __post_init__(self) -> None:
        for name in (
                "rank", "tensor_parallel_rank", "pipeline_parallel_rank",
                "expert_parallel_rank", "layer_start", "layer_end",
                "embedding_bytes", "decoder_layer_bytes",
                "mtp_bytes", "final_norm_bytes", "lm_head_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"model residency {name} must be an integer")
            if value < 0:
                raise ValueError(f"model residency {name} cannot be negative")
        if self.layer_end <= self.layer_start:
            raise ValueError("model residency layer range must be non-empty")

    @property
    def weight_resident_bytes(self) -> int:
        return (
            self.embedding_bytes
            + self.decoder_layer_bytes
            + self.mtp_bytes
            + self.final_norm_bytes
            + self.lm_head_bytes
        )


def _require_divisible(value: int, divisor: int, label: str) -> int:
    if value % divisor:
        raise NotImplementedError(
            f"analytical model residency requires {label}={value} to be "
            f"divisible by {divisor}")
    return value // divisor


def _attention_parameter_count(model: ModelConfig, tp_size: int) -> int:
    if model.attention_layout != "standard_gqa":
        raise NotImplementedError(
            "analytical model residency currently supports standard GQA; "
            f"got {model.attention_layout!r}")
    if model.attention_bias:
        raise NotImplementedError(
            "analytical model residency does not yet have a source-backed "
            "attention-bias placement contract")
    query_heads = _require_divisible(
        model.num_attention_heads, tp_size, "num_attention_heads")
    query_width = query_heads * model.head_dim
    kv_width = (
        model.kv_heads_per_tensor_parallel_rank(tp_size) * model.head_dim)
    projections = model.hidden_size * (2 * query_width + 2 * kv_width)
    qk_norm = 2 * model.head_dim if model.qk_norm else 0
    return projections + qk_norm


def _linear_weight_bytes(
    model: ModelConfig,
    *,
    out_features: int,
    in_features: int,
    module_path: str,
    module_type: str = "Linear",
) -> int:
    """Return source-defined resident storage for one local linear shard."""

    for name, value in (
            ("out_features", out_features), ("in_features", in_features)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"target residency {name} must be positive")
    dtype = model.weight_dtype_for_module(module_path, module_type)
    parameters = out_features * in_features
    if dtype in ("bf16", "fp16"):
        return parameters * model.dtype_bytes
    if dtype == "fp8":
        return parameters
    if dtype == "int4":
        group_size = model.weight_group_size
        if in_features % group_size:
            raise NotImplementedError(
                "INT4 residency requires input features divisible by the "
                "weight group size")
        if parameters % 2:
            raise NotImplementedError(
                "INT4 residency requires an even packed parameter count")
        packed_weights = parameters // 2
        bf16_group_scales = out_features * (in_features // group_size) * 2
        return packed_weights + bf16_group_scales
    raise NotImplementedError(
        f"unsupported target resident weight dtype: {dtype!r}")


def _target_attention_weight_bytes(
    model: ModelConfig,
    tp_size: int,
    layer_index: int,
    indexer_type: str,
) -> int:
    """Pinned vLLM MLA/DSA attention weights resident on one TP rank."""

    local_heads = _require_divisible(
        model.num_attention_heads, tp_size, "num_attention_heads")
    prefix = f"model.layers.{layer_index}.self_attn"
    total = _linear_weight_bytes(
        model,
        out_features=(
            model.q_lora_rank + model.kv_lora_rank
            + model.qk_rope_head_dim),
        in_features=model.hidden_size,
        module_path=f"{prefix}.fused_qkv_a_proj",
    )
    total += _linear_weight_bytes(
        model,
        out_features=local_heads * model.head_dim,
        in_features=model.q_lora_rank,
        module_path=f"{prefix}.q_b_proj",
    )
    total += _linear_weight_bytes(
        model,
        out_features=local_heads * (
            model.qk_nope_head_dim + model.v_head_dim),
        in_features=model.kv_lora_rank,
        module_path=f"{prefix}.kv_b_proj",
    )
    total += _linear_weight_bytes(
        model,
        out_features=model.hidden_size,
        in_features=local_heads * model.v_head_dim,
        module_path=f"{prefix}.o_proj",
    )
    # Decoder RMSNorms plus the latent Q/KV RMSNorms are TP replicated.
    total += (
        2 * model.hidden_size + model.q_lora_rank + model.kv_lora_rank
    ) * model.dtype_bytes
    if indexer_type not in ("none", "full", "shared"):
        raise ValueError(
            f"unknown target attention indexer type: {indexer_type!r}")
    if model.attention_layout == "dsa_mla" and indexer_type == "full":
        indexer = f"{prefix}.indexer"
        total += _linear_weight_bytes(
            model,
            out_features=model.index_n_heads * model.index_head_dim,
            in_features=model.q_lora_rank,
            module_path=f"{indexer}.wq_b",
        )
        total += _linear_weight_bytes(
            model,
            out_features=model.index_head_dim + model.index_n_heads,
            in_features=model.hidden_size,
            module_path=f"{indexer}.wk_weights_proj",
        )
        # torch LayerNorm has affine weight and bias.
        total += 2 * model.index_head_dim * model.dtype_bytes
    return total


def _target_dense_mlp_weight_bytes(
    model: ModelConfig,
    tp_size: int,
    layer_index: int,
) -> int:
    local_intermediate = _require_divisible(
        model.intermediate_size, tp_size, "intermediate_size")
    prefix = f"model.layers.{layer_index}.mlp"
    return (
        _linear_weight_bytes(
            model,
            out_features=2 * local_intermediate,
            in_features=model.hidden_size,
            module_path=f"{prefix}.gate_proj",
        )
        + _linear_weight_bytes(
            model,
            out_features=model.hidden_size,
            in_features=local_intermediate,
            module_path=f"{prefix}.down_proj",
        )
    )


def _target_moe_weight_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    layer_index: int,
) -> int:
    tp_size = parallelism.tensor_parallel_size
    ep_size = parallelism.expert_parallel_size
    if parallelism.enable_expert_parallel and ep_size != tp_size:
        raise NotImplementedError(
            "target MoE residency requires the pinned vLLM EP=TP placement")
    local_experts = _require_divisible(
        model.num_experts, ep_size, "num_experts")
    prefix = f"model.layers.{layer_index}.mlp"
    routed = 0
    for expert_index in range(local_experts):
        expert = f"{prefix}.experts.{expert_index}"
        routed += _linear_weight_bytes(
            model,
            out_features=2 * model.moe_intermediate_size,
            in_features=model.hidden_size,
            module_path=f"{expert}.gate_proj",
        )
        routed += _linear_weight_bytes(
            model,
            out_features=model.hidden_size,
            in_features=model.moe_intermediate_size,
            module_path=f"{expert}.down_proj",
        )
    shared_intermediate = (
        model.moe_intermediate_size * model.num_shared_experts)
    local_shared = _require_divisible(
        shared_intermediate, tp_size, "shared expert intermediate_size")
    shared = f"{prefix}.shared_experts"
    shared_bytes = (
        _linear_weight_bytes(
            model,
            out_features=2 * local_shared,
            in_features=model.hidden_size,
            module_path=f"{shared}.gate_proj",
        )
        + _linear_weight_bytes(
            model,
            out_features=model.hidden_size,
            in_features=local_shared,
            module_path=f"{shared}.down_proj",
        )
    )
    router_bytes = _linear_weight_bytes(
        model,
        out_features=model.num_experts,
        in_features=model.hidden_size,
        module_path=f"{prefix}.gate",
        module_type="GateLinear",
    )
    if model.target_semantics is None:
        raise AssertionError("target MoE residency lost target semantics")
    if model.target_semantics.moe_topk_method == "noaux_tc":
        correction_bias_bytes = model.num_experts * 4
    elif model.target_semantics.moe_topk_method == "group_limited_greedy":
        correction_bias_bytes = 0
    else:
        raise AssertionError(
            "unhandled target MoE top-k method: "
            f"{model.target_semantics.moe_topk_method!r}")
    # EP=TP keeps each selected expert rank-local.  In the aggregate H200
    # TP8/EP1 layout, every rank holds every expert but each expert's linear
    # weights are tensor-parallel shards; counting the full routed tensor on
    # every rank would turn the 8-way shard into an impossible 8x residency.
    if parallelism.enable_expert_parallel:
        routed_resident = routed
    else:
        routed_resident = _require_divisible(
            routed, tp_size, "non-EP routed MoE bytes")
    return routed_resident + shared_bytes + router_bytes + correction_bias_bytes


def _target_decoder_layer_weight_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    layer_index: int,
) -> int:
    if model.attention_layout == "dsa_mla":
        indexer_type = model.indexer_types[layer_index]
    elif model.attention_layout == "mla":
        indexer_type = "none"
    else:
        raise ValueError(
            "target decoder residency requires MLA or DSA-MLA attention")
    total = _target_attention_weight_bytes(
        model,
        parallelism.tensor_parallel_size,
        layer_index,
        indexer_type,
    )
    if model.layer_uses_moe(layer_index):
        return total + _target_moe_weight_bytes(
            model, parallelism, layer_index)
    return total + _target_dense_mlp_weight_bytes(
        model, parallelism.tensor_parallel_size, layer_index)


def _target_mtp_weight_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
) -> int:
    """Return source-defined target draft weights on one last-stage rank."""

    if model.model_type == "glm_moe_dsa":
        if (model.mtp_architecture != "glm_moe_dsa_mtp"
                or model.num_mtp_layers != 1):
            raise NotImplementedError(
                "target MTP residency supports the single GLM-5.2 draft layer")
        indexer_type = "full"
    elif model.model_type == "deepseek_v3":
        if (model.mtp_architecture != "deepseek_mtp"
                or model.num_mtp_layers != 1):
            raise NotImplementedError(
                "target MTP residency supports the single DeepSeek-V3 "
                "NextN draft layer")
        # DeepSeek-V3 has no DSA indexer; the NextN layer is a plain-MLA
        # MoE decoder layer plus eh_proj and the shared embedding/head.
        indexer_type = "none"
    else:
        raise NotImplementedError(
            "target MTP residency supports a single source-backed GLM-5.2 "
            "or DeepSeek-V3 draft layer")
    if not model.mtp_layer_uses_moe(0):
        raise NotImplementedError(
            "target MTP residency requires its source-defined MoE layer")
    tp_size = parallelism.tensor_parallel_size
    layer_index = model.num_hidden_layers
    local_vocab = _require_divisible(
        padded_vocab_size(model.vocab_size), tp_size, "padded_vocab_size")
    decoder_bytes = _target_attention_weight_bytes(
        model, tp_size, layer_index, indexer_type)
    decoder_bytes += _target_moe_weight_bytes(
        model, parallelism, layer_index)
    merge_bytes = _linear_weight_bytes(
        model,
        out_features=model.hidden_size,
        in_features=2 * model.hidden_size,
        module_path=f"model.layers.{layer_index}.eh_proj",
    )
    draft_embedding_bytes = local_vocab * model.hidden_size * model.dtype_bytes
    draft_head_bytes = _linear_weight_bytes(
        model,
        out_features=local_vocab,
        in_features=model.hidden_size,
        module_path=f"model.layers.{layer_index}.shared_head.head",
    )
    # enorm, hnorm, and shared_head.norm are replicated on every TP rank.
    draft_norm_bytes = 3 * model.hidden_size * model.dtype_bytes
    return (
        decoder_bytes
        + merge_bytes
        + draft_embedding_bytes
        + draft_head_bytes
        + draft_norm_bytes
    )


def _dense_mlp_parameter_count(model: ModelConfig, tp_size: int) -> int:
    if model.mlp_bias:
        raise NotImplementedError(
            "analytical model residency does not yet have a source-backed "
            "MLP-bias placement contract")
    local_intermediate = _require_divisible(
        model.intermediate_size, tp_size, "intermediate_size")
    return 3 * model.hidden_size * local_intermediate


def _moe_parameter_count(
    model: ModelConfig,
    parallelism: ParallelConfig,
) -> int:
    if model.model_type != "qwen3_moe" or model.num_shared_experts != 0:
        raise NotImplementedError(
            "analytical MoE model residency currently supports Qwen3MoE "
            "without shared experts")
    if (parallelism.tensor_parallel_size > model.num_experts
            or (not parallelism.enable_expert_parallel
                and model.moe_intermediate_size
                % parallelism.tensor_parallel_size)):
        raise NotImplementedError(
            "analytical Qwen3MoE residency requires TP <= num_experts and "
            "non-EP moe_intermediate_size divisible by TP")
    router = model.hidden_size * model.num_experts
    one_expert = 3 * model.hidden_size * model.moe_intermediate_size
    if parallelism.enable_expert_parallel:
        local_experts = _require_divisible(
            model.num_experts,
            parallelism.expert_parallel_size,
            "num_experts",
        )
        routed_experts = local_experts * one_expert
    else:
        routed_experts = _require_divisible(
            model.num_experts * one_expert,
            parallelism.tensor_parallel_size,
            "unsharded routed-expert parameters",
        )
    return router + routed_experts


def _decoder_layer_parameter_count(
    model: ModelConfig,
    parallelism: ParallelConfig,
    layer_index: int,
) -> int:
    tp_size = parallelism.tensor_parallel_size
    # Input and post-attention RMSNorm weights are replicated on every TP rank.
    parameters = 2 * model.hidden_size
    parameters += _attention_parameter_count(model, tp_size)
    if model.layer_uses_moe(layer_index):
        parameters += _moe_parameter_count(model, parallelism)
    else:
        parameters += _dense_mlp_parameter_count(model, tp_size)
    return parameters


def _attention_layer_parameter_count(
    model: ModelConfig,
    parallelism: ParallelConfig,
) -> int:
    # The input RMSNorm and attention projections execute on the AF Attention
    # pool. The post-attention RMSNorm and MLP weights reside on its FFN peer.
    return model.hidden_size + _attention_parameter_count(
        model, parallelism.tensor_parallel_size)


def model_rank_residencies(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    worker_role: str,
    include_mtp: bool = False,
) -> tuple[ModelRankResidency, ...]:
    """Return exact static weight bytes for every rank in one scheduler pool."""

    if not isinstance(model, ModelConfig):
        raise TypeError("model residency requires ModelConfig")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("model residency requires ParallelConfig")
    if not isinstance(include_mtp, bool):
        raise TypeError("model residency include_mtp must be boolean")
    whole_model_roles = {"colocated", "prefill", "decode"}
    attention_roles = {"attention", "prefill-attention", "decode-attention"}
    if worker_role not in whole_model_roles | attention_roles:
        raise NotImplementedError(
            "analytical KV capacity requires a whole-model or AF Attention "
            f"scheduler pool, got worker role {worker_role!r}")
    if not parallelism.is_rank_local:
        raise NotImplementedError(
            "analytical KV capacity currently supports DP=CP=SP=1")
    if model.compute_dtype not in ("bf16", "fp16"):
        raise NotImplementedError(
            "analytical model residency currently supports BF16 or FP16")
    if (model.model_type not in ("kimi_k2", "glm_moe_dsa", "deepseek_v3")
            and model.weight_dtype != model.compute_dtype):
        raise NotImplementedError(
            "analytical non-target model residency requires unquantized "
            "weights")
    if (model.num_mtp_layers
            and model.model_type not in ("kimi_k2", "glm_moe_dsa", "deepseek_v3")):
        raise NotImplementedError(
            "analytical model residency requires a complete MTP draft-weight "
            "placement contract")
    if model.model_type not in (
            "qwen3", "llama", "qwen3_moe", "kimi_k2", "glm_moe_dsa",
            "deepseek_v3"):
        raise NotImplementedError(
            "analytical model residency currently supports Qwen3, Qwen3MoE, "
            "Llama, Kimi-K2.6, GLM-5.2, and DeepSeek-V3")
    tp_size = parallelism.tensor_parallel_size
    pp_size = parallelism.pipeline_parallel_size
    layer_ranges = pipeline_layer_ranges_for_model(model, pp_size)
    if model.model_type == "qwen3_moe" and pp_size != 1:
        raise NotImplementedError(
            "analytical Qwen3MoE residency does not yet model pinned-vLLM "
            "embedding, final-norm, and LM-head replication across PP ranks")
    if model.tie_word_embeddings and pp_size > 1:
        raise NotImplementedError(
            "tied embeddings across pipeline ranks require an explicit "
            "cross-rank storage contract")
    local_vocab = _require_divisible(
        padded_vocab_size(model.vocab_size), tp_size, "padded_vocab_size")
    embedding_parameters = local_vocab * model.hidden_size
    element_bytes = model.dtype_bytes
    residencies = []
    target_model = model.model_type in ("kimi_k2", "glm_moe_dsa", "deepseek_v3")
    if include_mtp:
        if worker_role not in whole_model_roles:
            raise NotImplementedError(
                "MTP residency requires a whole-model scheduler pool")
        mtp_bytes = _target_mtp_weight_bytes(model, parallelism)
    else:
        mtp_bytes = 0
    for pp_rank, (layer_start, layer_end) in enumerate(layer_ranges):
        if worker_role in whole_model_roles:
            if target_model:
                decoder_bytes = sum(
                    _target_decoder_layer_weight_bytes(
                        model, parallelism, layer_index)
                    for layer_index in range(layer_start, layer_end)
                )
                decoder_parameters = None
            else:
                decoder_parameters = sum(
                    _decoder_layer_parameter_count(
                        model, parallelism, layer_index)
                    for layer_index in range(layer_start, layer_end)
                )
                decoder_bytes = decoder_parameters * element_bytes
        elif worker_role in attention_roles:
            if target_model:
                raise NotImplementedError(
                    "target analytical residency does not support AF splitting")
            decoder_parameters = sum(
                _attention_layer_parameter_count(model, parallelism)
                for unused_layer_index in range(layer_start, layer_end)
            )
            decoder_bytes = decoder_parameters * element_bytes
        else:
            raise AssertionError("validated worker role disappeared")
        for tp_rank in range(tp_size):
            rank = pp_rank * tp_size + tp_rank
            embedding_bytes = (
                embedding_parameters * element_bytes if pp_rank == 0 else 0)
            final_norm_bytes = (
                model.hidden_size * element_bytes
                if pp_rank == pp_size - 1 else 0)
            lm_head_bytes = (
                embedding_parameters * element_bytes
                if pp_rank == pp_size - 1
                and not model.tie_word_embeddings else 0)
            residencies.append(ModelRankResidency(
                rank=rank,
                tensor_parallel_rank=tp_rank,
                pipeline_parallel_rank=pp_rank,
                expert_parallel_rank=(
                    tp_rank if parallelism.enable_expert_parallel else 0),
                layer_start=layer_start,
                layer_end=layer_end,
                embedding_bytes=embedding_bytes,
                decoder_layer_bytes=decoder_bytes,
                mtp_bytes=(mtp_bytes if pp_rank == pp_size - 1 else 0),
                final_norm_bytes=final_norm_bytes,
                lm_head_bytes=lm_head_bytes,
            ))
    return tuple(residencies)
