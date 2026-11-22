"""Operator-group Roofline for the frozen V3, K2, Qwen and Llama sources."""

from dataclasses import dataclass
from math import expm1, log1p


MODEL_FAMILIES = {
    "deepseek-v3": "v3",
    "deepseek-v3-l32": "v3",
    "kimi-k2.6": "k2",
    "kimi-k2.7-code": "k2",
    "qwen3-30b-a3b": "gqa",
    "qwen3-32b": "gqa",
    "qwen3-32b-fp16": "gqa",
    "qwen3-8b": "gqa",
    "qwen3-4b": "gqa",
    "llama-3.1-8b-instruct": "gqa",
}

BF16_BYTES = 2
FP8_BLOCK_BYTES = 1 + 4 / 128**2


@dataclass(frozen=True, slots=True)
class FamilyLayout:
    family: str
    layer_types: tuple
    routed: int
    top_k: int
    linear_weight_bytes: float
    expert_weight_bytes: float


def family_layout(name, model, tp, dp, ep, pp):
    family = MODEL_FAMILIES[name]
    model = model.get("text_config", model)
    if (any(type(value) is not int or value < 1 for value in (tp, dp, ep, pp))
            or model["num_attention_heads"] % tp
            or pp > model["num_hidden_layers"]
            or ep not in (1, tp * dp) or dp > 1 and ep == 1):
        raise ValueError("invalid reviewed TP, DP, EP, PP layout")
    layers = model["num_hidden_layers"]
    if family in ("v3", "k2"):
        if model["model_type"] != ("deepseek_v3" if family == "v3" else "kimi_k2"):
            raise ValueError("MLA source model type does not match its checkpoint")
        if any(model[key] <= 0 for key in ("q_lora_rank", "kv_lora_rank", "qk_rope_head_dim",
                                           "qk_nope_head_dim", "v_head_dim")):
            raise ValueError("MLA model requires explicit positive latent dimensions")
        first_moe, frequency = model["first_k_dense_replace"], model.get("moe_layer_freq", 1)
        kinds = tuple("moe" if index >= first_moe and index % frequency == 0 else "dense"
                      for index in range(layers))
        routed, top_k = model["n_routed_experts"], model["num_experts_per_tok"]
        if family == "v3":
            if any(model[key] % tp for key in ("hidden_size", "intermediate_size", "moe_intermediate_size")):
                raise ValueError("V3 TP requires exact hidden and FFN dimensions")
            heads = model["num_attention_heads"] // tp
            block_widths = (model["hidden_size"], model["q_lora_rank"], model["kv_lora_rank"],
                            model["intermediate_size"] // tp, model["moe_intermediate_size"] // tp,
                            heads * (model["qk_nope_head_dim"] + model["qk_rope_head_dim"]),
                            heads * (model["qk_nope_head_dim"] + model["v_head_dim"]), heads * model["v_head_dim"])
            if any(width % 128 for width in block_widths):
                raise ValueError("V3 TP shards must preserve complete FP8 scale blocks")
            linear_dtype = expert_dtype = FP8_BLOCK_BYTES
        else:
            # K2's checkpoint ignores attention, dense and shared FFN in INT4
            # quantization. Marlin's routed weights have BF16 group scales.
            grouping = model["quantization_config"]["config_groups"]["group_0"]["weights"]
            if grouping["num_bits"] != 4 or not grouping["symmetric"] or grouping["group_size"] <= 0:
                raise ValueError("expected the pinned Kimi INT4 expert weight encoding")
            linear_dtype, expert_dtype = BF16_BYTES, .5 + 2 / grouping["group_size"]
    else:
        model_type = model["model_type"]
        expected = "llama" if name.startswith("llama-") else "qwen3_moe" if name == "qwen3-30b-a3b" else "qwen3"
        if model_type != expected:
            raise ValueError("GQA source model type does not match its checkpoint")
        kv_heads = model["num_key_value_heads"]
        if kv_heads <= 0 or max(kv_heads, tp) % min(kv_heads, tp):
            raise ValueError("KV head sharding or replication must be exact")
        routed = model["num_experts"] if model_type == "qwen3_moe" else 0
        top_k = model["num_experts_per_tok"] if routed else 0
        sparse_step = model.get("decoder_sparse_step", 1)
        dense_layers = frozenset(model.get("mlp_only_layers", ()))
        kinds = tuple("moe" if routed and index not in dense_layers and (index + 1) % sparse_step == 0
                      else "dense" for index in range(layers))
        # H200 FP16 and BF16 have the same dense rate and two-byte storage.
        linear_dtype = expert_dtype = BF16_BYTES
    if (not 0 <= top_k <= routed or routed and (not top_k or routed % ep)
            or not routed and ep != 1
            or "dense" in kinds and model["intermediate_size"] % tp
            or "moe" in kinds and ep == 1 and model["moe_intermediate_size"] % tp):
        raise ValueError("invalid expert ownership or feed-forward sharding")
    return FamilyLayout(family, kinds, routed, top_k, linear_dtype, expert_dtype)


def _query_pairs(query, history):
    return query * history + query * (query + 1) // 2


def _linear(hardware, weights, n, storage, *, operands=0, active=None):
    return hardware.roofline(2 * n * (weights if active is None else active),
                             weights * storage + operands, fp8=storage == FP8_BLOCK_BYTES)


def attention(family, model, tp, hardware, n, rows):
    """Use actual token/KV sizes, grouping the native projection and attention work."""
    h = model["hidden_size"]
    local_heads = model["num_attention_heads"] // tp
    if family == "gqa":
        dim = model.get("head_dim", h // model["num_attention_heads"])
        local_kv_heads = max(1, model["num_key_value_heads"] // tp)
        q_width, kv_width = local_heads * dim, local_kv_heads * dim
        projection_weights = h * (2 * q_width + 2 * kv_width)
        qkv_bytes = n * (2 * h + q_width + 2 * kv_width) * BF16_BYTES
        projection = _linear(hardware, projection_weights, n, BF16_BYTES, operands=qkv_bytes)
        visible = sum(_query_pairs(row.query, row.history) for row in rows)
        cache_rows = sum(row.query + row.history for row in rows)
        traffic = 2 * (kv_width * (cache_rows + n) + q_width * n) * BF16_BYTES
        attn = hardware.roofline(4 * visible * q_width, traffic)
        normalized = 2 * h + (q_width + kv_width if model["model_type"] != "llama" else 0)
        aux = hardware.roofline(0, 2 * n * (normalized + q_width + kv_width) * BF16_BYTES)
        return projection + aux, attn, 0.

    latent, rope = model["kv_lora_rank"], model["qk_rope_head_dim"]
    nope, value, q_rank = model["qk_nope_head_dim"], model["v_head_dim"], model["q_lora_rank"]
    storage = FP8_BLOCK_BYTES if family == "v3" else BF16_BYTES
    projection_weights = h * (q_rank + latent + rope) + q_rank * local_heads * (nope + rope) + h * local_heads * value
    projection = _linear(hardware, projection_weights, n, storage,
                         operands=2 * n * (h + q_rank + latent + rope) * BF16_BYTES)
    attn = hardware.roofline(0, n * (latent + rope) * BF16_BYTES)
    decode_rows = tuple(row for row in rows if row.phase == "decode")
    prefill_rows = tuple(row for row in rows if row.phase == "prefill")
    if decode_rows:
        decoded = sum(row.query for row in decode_rows)
        visible = sum(_query_pairs(row.query, row.history) for row in decode_rows)
        cache_rows = sum(row.query + row.history for row in decode_rows)
        projection += _linear(hardware, local_heads * latent * (nope + value), decoded, BF16_BYTES)
        attn += hardware.roofline(2 * visible * local_heads * (2 * latent + rope),
            BF16_BYTES * (cache_rows * (latent + rope) + decoded * local_heads * (2 * latent + rope)))
    if prefill_rows:
        queries = sum(row.query for row in prefill_rows)
        cache_rows = sum(row.history + row.query for row in prefill_rows)
        visible = sum(_query_pairs(row.query, row.history) for row in prefill_rows)
        projection += _linear(hardware, local_heads * latent * (nope + value), cache_rows, storage)
        attn += hardware.roofline(2 * visible * local_heads * (nope + rope + value),
            BF16_BYTES * (local_heads * (queries + cache_rows) * (nope + rope + value) + cache_rows * (latent + rope)))
    return projection, attn, 0.


def feed_forward(layout, model, tp, dp, ep, pp, hardware, n, total, dense, dp1_peak, *, split=False):
    """Query-only FFN plus a stage's mean activated expert weight and token work."""
    h = model["hidden_size"]
    if dense:
        estimate = (_linear(hardware, 3 * h * model["intermediate_size"] // tp, n, layout.linear_weight_bytes,
                            operands=4 * n * h * BF16_BYTES) if n else 0.)
        return (estimate, 0., 0.) if split else estimate

    intermediate = model["moe_intermediate_size"]
    weight_per_expert = 3 * h * (intermediate // tp if ep == 1 else intermediate)
    top_k, experts = layout.top_k, layout.routed
    live = (experts if top_k == experts else -experts * expm1(total * log1p(-top_k / experts))) if total else 0.
    owned_active = live / ep
    assigned = total * top_k / ep
    if total == 1 and dp1_peak:
        owned_active = assigned = dp1_peak
    estimate = _linear(hardware, weight_per_expert * owned_active, assigned, layout.expert_weight_bytes,
                       active=weight_per_expert, operands=4 * assigned * h * BF16_BYTES)
    if not n:
        return (estimate, 0., 0.) if split else estimate
    moe_sp = dp > 1 and tp > 1
    local_n = (n + tp - 1) // tp if moe_sp else n
    expert_reduction = 1 if moe_sp else tp
    route = _linear(hardware, h * experts, local_n, BF16_BYTES, operands=4 * local_n * experts)
    estimate += route
    shared = model.get("n_shared_experts", 0) or 0
    if shared:
        estimate += _linear(hardware, 3 * h * intermediate * shared // expert_reduction,
                            local_n, layout.linear_weight_bytes)
    return (estimate, route, 0.) if split else estimate
