"""Shape inference for transformer GEMM layers.

This module converts model semantics into GEMM dimensions before workload
lowering. It has no dependency on profiler traces or predictor implementations.

Layer positions:
  qkv_proj    : Q+K+V fused projection with sharded Q and sharded/replicated KV
  fused_qkv_a_proj: vLLM MLA/DSA fused Q-A and KV-A projection (replicated)
  q_a_proj    : Transformers GLM DSA latent-Q compression (replicated)
  q_b_proj    : MLA/DSA latent-Q expansion, column-parallel
  kv_a_proj_with_mqa: Transformers GLM DSA latent-KV compression (replicated)
  kv_b_proj   : MLA/DSA latent-KV expansion, column-parallel
  indexer_wq_b: GLM DSA indexer query expansion (replicated)
  indexer_wk  : GLM DSA indexer key projection (replicated)
  indexer_weights_proj: GLM DSA indexer head weights (replicated)
  indexer_wk_weights_proj: vLLM GLM DSA fused indexer key/weight projection
  o_proj      : output projection, row-parallel
  gate_up_proj: fused gate+up (SwiGLU), column-parallel
  down_proj   : down projection, row-parallel
  lm_head     : vLLM padded vocabulary projection, sharded over TP ranks
"""

from .model_config import ModelConfig


VLLM_VOCAB_PADDING_SIZE = 64


def padded_vocab_size(vocab_size: int) -> int:
    """Apply the pinned vLLM 0.11.0 vocabulary padding contract."""

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return (
        (vocab_size + VLLM_VOCAB_PADDING_SIZE - 1)
        // VLLM_VOCAB_PADDING_SIZE
        * VLLM_VOCAB_PADDING_SIZE
    )


def infer_gemm_dimensions(model: ModelConfig, layer_position: str,
                           seq_len: int, batch_size: int = 1,
                           tp_size: int = 1):
    """Return (M, N, K) for a named GEMM position given model config.

    TP=N sharding:
      qkv_proj: Q heads are sharded; KV heads are sharded or replicated
      Column-parallel (gate_up_proj): N divided by tp_size
      Row-parallel    (o_proj, down_proj):      K divided by tp_size
      lm_head: padded vocabulary N divided by tp_size

    Args:
        model: ModelConfig from load_model_config()
        layer_position: one of qkv_proj / fused_qkv_a_proj / q_a_proj /
                        q_b_proj / kv_a_proj_with_mqa / kv_b_proj /
                        indexer_wq_b / indexer_wk / indexer_weights_proj /
                        o_proj / gate_up_proj / down_proj / lm_head
        seq_len: sequence length
        batch_size: batch size
        tp_size: tensor parallelism factor

    Returns:
        (M, N, K) tuple of integers

    Raises:
        ValueError: if tp_size does not divide the sharded dimension, or layer
                    position is unknown.
    """
    if isinstance(tp_size, bool) or not isinstance(tp_size, int):
        raise TypeError("tp_size must be an integer")
    if tp_size < 1:
        raise ValueError("tp_size must be positive")

    H = model.hidden_size
    I = model.intermediate_size
    HQ = model.num_attention_heads
    D = model.head_dim
    M = batch_size * seq_len

    if layer_position == "qkv_proj":
        if HQ % tp_size:
            raise ValueError(
                "qkv_proj query heads must be divisible by tp_size: "
                f"{HQ} vs {tp_size}")
        local_query_heads = HQ // tp_size
        local_kv_heads = model.kv_heads_per_tensor_parallel_rank(tp_size)
        N = (local_query_heads + 2 * local_kv_heads) * D
        K = H
    elif layer_position == "fused_qkv_a_proj":
        if model.attention_layout not in ("mla", "dsa_mla"):
            raise ValueError(
                f"{layer_position} requires the pinned MLA/DSA vLLM "
                "execution contract")
        # Both pinned vLLM implementations use disable_tp=True for the merged
        # Q-A and KV-A projection, so every TP rank sees the full output width.
        N = (
            model.q_lora_rank
            + model.kv_lora_rank
            + model.qk_rope_head_dim
        )
        K = H
    elif layer_position in (
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
            "indexer_wq_b", "indexer_wk", "indexer_weights_proj",
            "indexer_wk_weights_proj"):
        if model.attention_layout not in ("mla", "dsa_mla"):
            raise ValueError(
                f"{layer_position} requires an MLA or DSA model")
        if layer_position in (
                "q_a_proj", "kv_a_proj_with_mqa", "indexer_wq_b",
                "indexer_wk", "indexer_weights_proj",
                "indexer_wk_weights_proj") and (
                    model.attention_layout != "dsa_mla"):
            raise ValueError(
                f"{layer_position} requires the pinned GLM DSA execution "
                "contract")
        if layer_position == "q_a_proj":
            N = model.q_lora_rank
            K = H
        elif layer_position == "q_b_proj":
            total_n = HQ * (
                model.qk_nope_head_dim + model.qk_rope_head_dim)
            if total_n % tp_size:
                raise ValueError(
                    "MLA q_b_proj output must be divisible by tp_size: "
                    f"{total_n} vs {tp_size}")
            N = total_n // tp_size
            K = model.q_lora_rank
        elif layer_position == "kv_a_proj_with_mqa":
            N = model.kv_lora_rank + model.qk_rope_head_dim
            K = H
        elif layer_position == "kv_b_proj":
            total_n = HQ * (
                model.qk_nope_head_dim + model.v_head_dim)
            if total_n % tp_size:
                raise ValueError(
                    "MLA kv_b_proj output must be divisible by tp_size: "
                    f"{total_n} vs {tp_size}")
            N = total_n // tp_size
            K = model.kv_lora_rank
        elif layer_position == "indexer_wq_b":
            N = model.index_n_heads * model.index_head_dim
            K = model.q_lora_rank
        elif layer_position == "indexer_wk":
            N = model.index_head_dim
            K = H
        elif layer_position == "indexer_weights_proj":
            N = model.index_n_heads
            K = H
        elif layer_position == "indexer_wk_weights_proj":
            N = model.index_head_dim + model.index_n_heads
            K = H
        else:
            raise AssertionError(
                f"unhandled MLA/DSA projection: {layer_position!r}")
    elif layer_position == "o_proj":
        if model.attention_layout == "standard_gqa":
            total_k = HQ * D
        elif model.attention_layout in ("mla", "dsa_mla"):
            total_k = HQ * model.v_head_dim
        else:
            raise AssertionError(
                f"unhandled attention layout: {model.attention_layout!r}")
        if total_k % tp_size != 0:
            raise ValueError(
                f"o_proj K={total_k} not divisible by tp_size={tp_size}")
        N = H
        K = total_k // tp_size
    elif layer_position == "gate_up_proj":
        total_n = 2 * I
        if total_n % tp_size != 0:
            raise ValueError(
                f"gate_up_proj N={total_n} not divisible by tp_size={tp_size}")
        N = total_n // tp_size
        K = H
    elif layer_position == "down_proj":
        if I % tp_size != 0:
            raise ValueError(
                f"down_proj K={I} not divisible by tp_size={tp_size}")
        N = H
        K = I // tp_size
    elif layer_position == "lm_head":
        padded_vocab = padded_vocab_size(model.vocab_size)
        if padded_vocab % tp_size != 0:
            raise ValueError(
                "padded lm_head vocabulary must be divisible by tp_size: "
                f"{padded_vocab} vs {tp_size}")
        N = padded_vocab // tp_size
        K = H
    elif layer_position == "router":
        if model.num_experts <= 0:
            raise ValueError("router position requires a MoE model")
        N = model.num_experts
        K = H
    else:
        raise ValueError(f"Unknown layer_position: '{layer_position}'")

    return M, N, K
