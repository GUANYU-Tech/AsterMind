"""Static, rank-local model-weight residency for KV capacity planning."""

from dataclasses import dataclass

from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.workload.model_config import (
    GLM_BLOCK_FP8_CONFIG_SHA256S, KIMI_K2_CONFIG_SHA256,
    KIMI_K2_7_CODE_CONFIG_SHA256, ModelConfig,
    is_validated_qwen3_moe_ep2_model_scope,
    is_validated_qwen3_pipeline_model_scope,
)
from src.serving.workload.shape_inference import padded_vocab_size


# Unique binding for the DeepGEMM dependency pinned by vLLM 0.30's
# cmake/external_projects/deepgemm.cmake. This is a source storage profile,
# not a declaration that the runtime selected or loaded this provider.
GLM_SM90_DEEP_GEMM_STORAGE_SOURCE = {
    "repository": "https://github.com/vllm-project/DeepGEMM",
    "revision": "e1f418c2a4f20818221f6b0e578b4c2f634d4c3f",
    "source_file": "csrc/apis/layout.hpp",
    "source_symbol": "transform_sf_into_required_layout",
    "scope": "arch_major=9, FP32 SFB, recipe=(1,128,128), is_sfa=false",
    "workspace_alignment_file": "csrc/jit_kernels/heuristics/runtime.hpp",
    "workspace_alignment_symbol": "HeuristicsRuntime::get_theoretical_mk_alignment_for_contiguous_layout",
}


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
    additional_parameter_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
                "rank", "tensor_parallel_rank", "pipeline_parallel_rank",
                "expert_parallel_rank", "layer_start", "layer_end",
                "embedding_bytes", "decoder_layer_bytes",
                "mtp_bytes", "final_norm_bytes", "lm_head_bytes", "additional_parameter_bytes"):
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
            + self.additional_parameter_bytes
        )


def _require_divisible(value: int, divisor: int, label: str) -> int:
    if value % divisor:
        raise NotImplementedError(
            f"analytical model residency requires {label}={value} to be "
            f"divisible by {divisor}")
    return value // divisor


def kimi_k25_outer_loader_payload_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    vision_tensor_parallel_mode: str,
) -> dict[str, int]:
    """Active-tower BF16 parameters for the exact K2.6 and K2.7-Code configs.

    vLLM 0.30 KimiK25Config/KimiK25VisionConfig retain the legacy vt_* keys
    as metadata, while the constructor defaults define H=1152, I=4304,
    27 layers and 16 heads. KimiK25Config replaces mm_hidden_size==H with
    the text hidden size. CompressedTensorsConfig is ignored for the tower
    and projector, so every parameter here is BF16, including biases/norms.
    This is an explicit enabled-tower, no-offload constructor scenario;
    no image input does not establish language_model_only/vision_chunk skip.
    Buffers, post-load storage, text weights, reserves and KV are excluded.
    The normal capacity entry still rejects the extracted text inventory.
    """
    if not isinstance(model, ModelConfig):
        raise TypeError("Kimi outer payload requires ModelConfig")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("Kimi outer payload requires ParallelConfig")
    if (model.model_type != "kimi_k2"
            or model.config_sha256 not in (KIMI_K2_CONFIG_SHA256, KIMI_K2_7_CODE_CONFIG_SHA256)
            or model.config_extraction_path != "text_config"
            or model.compute_dtype != "bf16" or model.hidden_size != 7168):
        raise NotImplementedError("Kimi outer payload requires the exact BF16 Kimi-K2.6 or Kimi-K2.7-Code parent config")
    if not parallelism.is_rank_local or parallelism.pipeline_parallel_size != 1:
        raise NotImplementedError("Kimi outer payload requires PP=DP=CP=SP=1")
    if vision_tensor_parallel_mode not in ("weights", "data"):
        raise ValueError("vision tensor parallel mode must be weights or data")
    # vision.is_vit_use_data_parallel forces DP when heads cannot split.
    tp = parallelism.tensor_parallel_size
    vision_tp = 1 if vision_tensor_parallel_mode == "data" or 16 % tp else tp
    h, intermediate, layers = 1152, 4304, 27
    local_h = _require_divisible(h, vision_tp, "vision hidden_size")
    local_i = _require_divisible(intermediate, vision_tp, "vision intermediate_size")
    entries: dict[str, int] = {}

    def add(path: str, elements: int) -> None:
        if path in entries:
            raise ValueError(f"duplicate Kimi outer loader parameter: {path}")
        entries[path] = 2 * elements

    add("vision_tower.patch_embed.proj.weight", h * 3 * 14 * 14)
    add("vision_tower.patch_embed.proj.bias", h)
    add("vision_tower.patch_embed.pos_emb.weight", 64 * 64 * h)
    for layer in range(layers):
        prefix = f"vision_tower.encoder.blocks.{layer}"
        for norm in ("norm0", "norm1"):
            add(f"{prefix}.{norm}.weight", h)
            add(f"{prefix}.{norm}.bias", h)
        add(f"{prefix}.wqkv.weight", 3 * local_h * h)
        add(f"{prefix}.wqkv.bias", 3 * local_h)
        add(f"{prefix}.wo.weight", h * local_h)
        add(f"{prefix}.wo.bias", h)
        add(f"{prefix}.mlp.fc0.weight", local_i * h)
        add(f"{prefix}.mlp.fc0.bias", local_i)
        add(f"{prefix}.mlp.fc1.weight", h * local_i)
        add(f"{prefix}.mlp.fc1.bias", h)
    for suffix in ("weight", "bias"):
        add(f"vision_tower.encoder.final_layernorm.{suffix}", h)
        add(f"mm_projector.pre_norm.{suffix}", h)
    # Patchmerger's two ReplicatedLinear modules never shard their parameters.
    merged = 2 * 2 * h
    add("mm_projector.linear_1.weight", merged * merged)
    add("mm_projector.linear_1.bias", merged)
    add("mm_projector.linear_2.weight", model.hidden_size * merged)
    add("mm_projector.linear_2.bias", model.hidden_size)
    return entries


def dsv4_routed_loader_payload_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
) -> dict[str, int]:
    """Return the V4 EP-off routed loader payload on one TP rank.

    This is the aligned Marlin source constructor inventory, before conversion;
    it cannot be supplied to the normal KV capacity allocator as residency.
    DeepseekV4MoE's non-MegaMoE FusedMoEFactory retains every expert with
    intermediate_size / TP. Mxfp4MoEMethod.create_weights registers packed
    uint8 gate/up and down matrices plus one uint8 UE8M0 scale per 32 input
    elements (vLLM 0.30 quantization/mxfp4.py). Converted scales, padding,
    workspaces, other model weights, MTP and runtime reserves are excluded.
    The MARLIN oracle rounds H to 256 and local I to 128; this subset rejects
    dimensions needing that padding instead of charging unpadded allocations.
    """
    if not isinstance(model, ModelConfig):
        raise TypeError("V4 loader payload requires ModelConfig")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("V4 loader payload requires ParallelConfig")
    if (model.model_type != "deepseek_v4" or model.expert_dtype != "fp4"
            or model.compute_dtype != "bf16"):
        raise NotImplementedError("V4 loader payload requires BF16/MXFP4 V4")
    if (not parallelism.is_rank_local
            or parallelism.pipeline_parallel_size != 1
            or parallelism.enable_expert_parallel):
        raise NotImplementedError(
            "V4 loader payload requires non-MegaMoE EP-off, PP=DP=CP=SP=1")
    local_intermediate = _require_divisible(
        model.moe_intermediate_size, parallelism.tensor_parallel_size,
        "moe_intermediate_size")
    # The outer DeepseekV4MoE metadata also asserts E divisible by TP.
    _require_divisible(model.num_experts, parallelism.tensor_parallel_size, "num_experts")
    _require_divisible(model.hidden_size, 256, "Marlin hidden_size")
    _require_divisible(local_intermediate, 128, "Marlin local moe_intermediate_size")
    # Refuse source floor division that would truncate a scale group.
    hidden_groups = _require_divisible(model.hidden_size, 32, "hidden_size")
    intermediate_groups = _require_divisible(
        local_intermediate, 32, "local moe_intermediate_size")
    experts = model.num_experts
    return {
        "experts_per_rank": experts,
        "intermediate_size_per_rank": local_intermediate,
        "layers": model.num_hidden_layers,
        "w13_packed_bytes_per_layer": (
            experts * 2 * local_intermediate * (model.hidden_size // 2)),
        "w2_packed_bytes_per_layer": (
            experts * model.hidden_size * (local_intermediate // 2)),
        "w13_ue8m0_bytes_per_layer": (
            experts * 2 * local_intermediate * hidden_groups),
        "w2_ue8m0_bytes_per_layer": (
            experts * model.hidden_size * intermediate_groups),
    }


def dsv4_non_routed_loader_payload_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    attention_backend: str,
) -> dict[str, int]:
    """Inventory target-off V4 non-routed constructor parameters per TP rank.

    The explicit FlashMLA/non-MegaMoE/EP-off scenario uses NVIDIA model.py,
    attention.py and compressor.py in vLLM 0.30. FP8 serialized linears use
    Fp8LinearMethod.create_weights: one byte per value and one UE8M0 byte per
    128x128 block. Compressors, index weights, gate, norms and vocabulary
    weights are BF16; APE/mHC/sinks/correction biases are explicitly FP32.
    Hash tables are replicated INT32. Values describe registered parameters
    before post-load conversion, not allocator-ready runtime residency. KV,
    rotary/topk buffers, broadcast-weight derivation, MTP and scratch/reserves
    are separate objects and are excluded from this parameter inventory.
    """
    routed = dsv4_routed_loader_payload_bytes(model, parallelism)
    if attention_backend != "flashmla":
        raise NotImplementedError("V4 parameter payload requires explicit FlashMLA")
    if model.weight_dtype != "fp8" or model.tie_word_embeddings:
        raise NotImplementedError("V4 parameter payload requires FP8 and separate vocabulary weights")
    if model.target_semantics is None or model.target_semantics.moe_topk_method != "noaux_tc":
        raise NotImplementedError("V4 parameter payload requires the text noaux_tc routing source")
    tp = parallelism.tensor_parallel_size
    h = model.hidden_size
    d = model.head_dim
    local_heads = _require_divisible(model.num_attention_heads, tp, "num_attention_heads")
    if local_heads > 128:
        raise NotImplementedError("V4 FlashMLA source supports at most 128 local heads")
    local_groups = _require_divisible(model.o_groups, tp, "o_groups")
    heads_per_group = _require_divisible(model.num_attention_heads, model.o_groups, "heads per output group")
    local_vocab = _require_divisible(padded_vocab_size(model.vocab_size), tp, "padded_vocab_size")
    entries: dict[str, int] = {}

    def add(path: str, elements: int, element_bytes: int) -> None:
        if path in entries:
            raise ValueError(f"duplicate V4 loader parameter: {path}")
        entries[path] = elements * element_bytes

    def fp8_linear(path: str, outputs: int, inputs: int) -> None:
        add(f"{path}.weight", outputs * inputs, 1)
        add(f"{path}.weight_scale_inv", ((outputs + 127) // 128) * ((inputs + 127) // 128), 1)

    def compressor(path: str, ratio: int, width: int) -> None:
        coff = 2 if ratio == 4 else 1
        add(f"{path}.fused_wkv_wgate.weight", 2 * coff * width * h, 2)
        add(f"{path}.norm.weight", width, 2)
        add(f"{path}.ape", ratio * coff * width, 4)

    add("model.embed_tokens.weight", local_vocab * h, 2)
    add("lm_head.weight", local_vocab * h, 2)
    add("model.norm.weight", h, 2)
    add("model.hc_head_fn", model.hc_mult * model.hc_mult * h, 4)
    add("model.hc_head_base", model.hc_mult, 4)
    add("model.hc_head_scale", 1, 4)
    mix = (2 + model.hc_mult) * model.hc_mult
    local_shared = routed["intermediate_size_per_rank"] * model.num_shared_experts
    for layer in range(model.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        attn = f"{prefix}.attn"
        ffn = f"{prefix}.ffn"
        add(f"{prefix}.attn_norm.weight", h, 2)
        add(f"{prefix}.ffn_norm.weight", h, 2)
        for branch in ("attn", "ffn"):
            add(f"{prefix}.hc_{branch}_fn", mix * model.hc_mult * h, 4)
            add(f"{prefix}.hc_{branch}_base", mix, 4)
            add(f"{prefix}.hc_{branch}_scale", 3, 4)
        fp8_linear(f"{attn}.fused_wqa_wkv", model.q_lora_rank + d, h)
        fp8_linear(f"{attn}.wq_b", local_heads * d, model.q_lora_rank)
        fp8_linear(f"{attn}.wo_a", local_groups * model.o_lora_rank, heads_per_group * d)
        fp8_linear(f"{attn}.wo_b", h, local_groups * model.o_lora_rank)
        add(f"{attn}.q_norm.weight", model.q_lora_rank, 2)
        add(f"{attn}.kv_norm.weight", d, 2)
        # FlashMLA.get_padded_num_q_heads defines 64/128 sink allocation.
        add(f"{attn}.attn_sink", 64 if local_heads <= 64 else 128, 4)
        ratio = model.kernel_compress_ratios[layer]
        if ratio > 1:
            compressor(f"{attn}.compressor", ratio, d)
        if ratio == 4:
            indexer = f"{attn}.indexer"
            fp8_linear(f"{indexer}.wq_b", model.index_n_heads * model.index_head_dim, model.q_lora_rank)
            add(f"{indexer}.weights_proj.weight", model.index_n_heads * h, 2)
            compressor(f"{indexer}.compressor", ratio, model.index_head_dim)
        add(f"{ffn}.gate.weight", model.num_experts * h, 2)
        if layer < model.num_hash_layers:
            add(f"{ffn}.gate.tid2eid", model.vocab_size * model.num_experts_per_tok, 4)
        else:
            # Validated text configs use topk_method=noaux_tc.
            add(f"{ffn}.gate.e_score_correction_bias", model.num_experts, 4)
        if local_shared:
            fp8_linear(f"{ffn}.shared_experts.gate_up_proj", 2 * local_shared, h)
            fp8_linear(f"{ffn}.shared_experts.down_proj", h, local_shared)
    return entries


def dsv4_postload_parameter_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    hardware_id: str,
    linear_backend: str,
    routed_backend: str,
) -> dict[str, int]:
    """Target-off registered parameter payload for a declared SM90 profile.

    Explicit CUDA DeepGEMM block-FP8 linears plus non-batched Marlin MXFP4
    routed experts, ordinary sharded vocabulary, FlashMLA and EP-off/PP1.
    This is source storage accounting, not runtime backend selection.

    fp8_utils.deepgemm_post_process_weight_scale_block upcasts checkpoint
    UE8M0 scales to FP32. The pinned SM90 SFB (128,128) layout is unchanged
    (same storage branch as GLM_SM90_DEEP_GEMM_STORAGE_SOURCE), including
    wo_a's grouped-BMM view. Marlin repack changes layout but retains packed
    4-bit payload; mxfp4_marlin_process_scales returns UE8M0 after its BF16
    intermediate, so that temporary must not double resident scale bytes.
    Replaced loader tensors are not retained parameters. Buffers, transient
    conversion peak, kernel workspaces and runtime reserves are excluded;
    this inventory alone cannot establish normal available-KV capacity.
    """
    if (hardware_id != "nvidia_h200" or linear_backend != "deep_gemm"
            or routed_backend != "marlin"):
        raise NotImplementedError("V4 post-load parameters require explicit H200 DeepGEMM/Marlin source storage")
    routed = dsv4_routed_loader_payload_bytes(model, parallelism)
    parameters = dsv4_non_routed_loader_payload_bytes(
        model, parallelism, attention_backend="flashmla")
    for path in parameters:
        if path.endswith(".weight_scale_inv"):
            parameters[path] *= 4
    for layer in range(model.num_hidden_layers):
        prefix = f"model.layers.{layer}.ffn.experts"
        parameters[f"{prefix}.w13_weight"] = routed["w13_packed_bytes_per_layer"]
        parameters[f"{prefix}.w2_weight"] = routed["w2_packed_bytes_per_layer"]
        parameters[f"{prefix}.w13_weight_scale"] = routed["w13_ue8m0_bytes_per_layer"]
        parameters[f"{prefix}.w2_weight_scale"] = routed["w2_ue8m0_bytes_per_layer"]
    return parameters


def dsv4_model_buffer_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    max_batched_tokens: int,
) -> dict[str, int]:
    """Unique V4 model-owned persistent tensors under the CUDA MTP-off source.

    The model owns one shared INT32 top-k buffer, not one per C4 indexer.
    finalize_mhc_broadcast_weights retains one FP32 first-layer reduction.
    get_rope caches the FP32 rotary keys actually used by target layers.
    Flash uses SWA and compressed; Pro MTP-off uses compressed only. C4/C128,
    compressor and indexer share these objects, not one per layer. The draft
    tail does not add an MTP-off key. Extent uses checkpoint limits, not the shorter
    Serving max_model_len. CUDA has no ROCm-only compressor scratch, and
    MTP-off has no target hidden buffer. Runtime/attention metadata, library
    state, KV storage, activations and reserves remain outside this scope.
    """
    # Reuse the complete dtype/parallelism/layout source boundary.
    dsv4_non_routed_loader_payload_bytes(model, parallelism, attention_backend="flashmla")
    if type(max_batched_tokens) is not int or max_batched_tokens <= 0:
        raise ValueError("V4 model buffers require positive integer max_batched_tokens")
    semantics = model.target_semantics
    if semantics is None or semantics.rope_type != "dsv4_yarn":
        raise NotImplementedError("V4 buffers require the declared V4 YaRN source")
    extent = semantics.rope_original_max_position_embeddings * semantics.rope_factor
    if int(extent) != extent:
        raise NotImplementedError("V4 compressed rotary cache extent must be integral")
    entries = {"model.topk_indices_buffer": max_batched_tokens * model.index_topk * 4,
        "model.layers.0.hc_attn_fn_broadcast": (2 + model.hc_mult) * model.hc_mult * model.hidden_size * 4}
    # kernel_compress_ratios includes the draft tail; MTP-off constructs
    # target layers only. Pro's sole SWA-only ratio is in that draft tail.
    target_ratios = model.kernel_compress_ratios[:model.num_hidden_layers]
    if any(ratio <= 1 for ratio in target_ratios):
        entries["model.shared_swa_rotary.cos_sin_cache"] = model.max_position_embeddings * model.qk_rope_head_dim * 4
    if any(ratio > 1 for ratio in target_ratios):
        entries["model.shared_compressed_rotary.cos_sin_cache"] = int(extent) * model.qk_rope_head_dim * 4
    return entries


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


def glm_block_fp8_loader_payload_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    include_mla_absorption: bool,
) -> dict[str, int]:
    """GLM-5.3 target-off parameter payload on one PP1 rank, not normal KV.

    Fp8LinearMethod/create_fp8_scale_parameter and Fp8MoEMethod.create_weights
    define FP8 values plus FP32 128x128 block scales. Exact BF16 exclusions
    and quant_config=None modules remain separate parameters. EP-off shards
    each expert's intermediate dimension; EP=TP shards expert ownership.

    include_mla_absorption declares the ordinary CUDA/CP1 MLA post-load
    branch: retain original FP8 kv_b_proj and add the disjoint BF16 W_UK_T
    and W_UV views of its dequantized storage once. No extra dequant tensor
    is charged on top of these views. Other backend conversions, persistent
    buffers, workspace/reserves, MTP and actual loader/provider binding are
    excluded. This inventory must not be used as allocator-ready residency.
    """
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("GLM FP8 payload requires ModelConfig and ParallelConfig")
    if not isinstance(include_mla_absorption, bool):
        raise TypeError("include_mla_absorption must be boolean")
    if (model.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S
            or model.model_type != "glm_moe_dsa" or model.weight_dtype != "fp8"
            or model.compute_dtype != "bf16" or model.tie_word_embeddings
            or model.quantized_weight_targets != ("Linear",)):
        raise NotImplementedError("GLM FP8 payload requires the exact official GLM-5.2-FP8/GLM-5.3 config")
    if not parallelism.is_rank_local or parallelism.pipeline_parallel_size != 1:
        raise NotImplementedError("GLM FP8 payload requires PP=DP=CP=SP=1")
    tp = parallelism.tensor_parallel_size
    ep = parallelism.expert_parallel_size
    if parallelism.enable_expert_parallel and ep != tp:
        raise NotImplementedError("GLM FP8 payload requires EP off or EP=TP")
    h = model.hidden_size
    heads = _require_divisible(model.num_attention_heads, tp, "attention heads")
    experts = _require_divisible(model.num_experts, ep, "routed experts")
    intermediate = _require_divisible(
        model.moe_intermediate_size, 1 if parallelism.enable_expert_parallel else tp,
        "routed intermediate")
    if h % 128 or intermediate % 128:
        raise NotImplementedError("GLM FP8 payload requires unrefined 128x128 expert scales")
    entries: dict[str, int] = {}

    def add(path: str, elements: int, element_bytes: int) -> None:
        if path in entries:
            raise ValueError(f"duplicate GLM FP8 payload parameter: {path}")
        entries[path] = elements * element_bytes

    def linear(path: str, outputs: int, inputs: int, *, quantized: bool = True) -> None:
        dtype = model.weight_dtype_for_module(path, "Linear") if quantized else "bf16"
        if dtype == "bf16":
            add(path + ".weight", outputs * inputs, 2)
        elif dtype == "fp8":
            add(path + ".weight", outputs * inputs, 1)
            add(path + ".weight_scale_inv", ((outputs + 127) // 128) * ((inputs + 127) // 128), 4)
        else:
            raise NotImplementedError(f"unsupported GLM payload linear dtype: {dtype}")

    vocab = _require_divisible(padded_vocab_size(model.vocab_size), tp, "vocabulary")
    add("model.embed_tokens.weight", vocab * h, 2)
    linear("lm_head", vocab, h)
    add("model.norm.weight", h, 2)
    for layer in range(model.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        attn = prefix + ".self_attn"
        linear(attn + ".fused_qkv_a_proj", model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim, h)
        linear(attn + ".q_b_proj", heads * model.head_dim, model.q_lora_rank)
        linear(attn + ".kv_b_proj", heads * (model.qk_nope_head_dim + model.v_head_dim), model.kv_lora_rank)
        linear(attn + ".o_proj", h, heads * model.v_head_dim)
        for name, width in (("q_a_layernorm", model.q_lora_rank), ("kv_a_layernorm", model.kv_lora_rank)):
            add(attn + "." + name + ".weight", width, 2)
        for name in ("input_layernorm", "post_attention_layernorm"):
            add(prefix + "." + name + ".weight", h, 2)
        if model.indexer_types[layer] == "full":
            indexer = attn + ".indexer"
            linear(indexer + ".wq_b", model.index_n_heads * model.index_head_dim, model.q_lora_rank)
            linear(indexer + ".wk_weights_proj", model.index_head_dim + model.index_n_heads, h, quantized=False)
            add(indexer + ".k_norm.weight", model.index_head_dim, 4)
            add(indexer + ".k_norm.bias", model.index_head_dim, 4)
        elif model.indexer_types[layer] != "shared":
            raise ValueError("GLM FP8 payload requires full/shared source indexer types")
        if include_mla_absorption:
            if model.weight_dtype_for_module(attn + ".kv_b_proj", "Linear") != "fp8":
                raise NotImplementedError("GLM absorption payload requires retained FP8 kv_b_proj")
            add(attn + ".W_UK_T", heads * model.qk_nope_head_dim * model.kv_lora_rank, 2)
            add(attn + ".W_UV", heads * model.kv_lora_rank * model.v_head_dim, 2)
        mlp = prefix + ".mlp"
        if model.layer_uses_moe(layer):
            routed = mlp + ".experts"
            if any(model.weight_dtype_for_module(routed + ".0." + name, "Linear") != "fp8"
                   for name in ("gate_proj", "up_proj", "down_proj")):
                raise NotImplementedError("GLM payload requires the block-FP8 routed-expert branch")
            add(routed + ".w13_weight", experts * 2 * intermediate * h, 1)
            add(routed + ".w2_weight", experts * h * intermediate, 1)
            add(routed + ".w13_weight_scale_inv", experts * 2 * (intermediate // 128) * (h // 128), 4)
            add(routed + ".w2_weight_scale_inv", experts * (h // 128) * (intermediate // 128), 4)
            linear(mlp + ".gate", model.num_experts, h, quantized=False)
            add(mlp + ".gate.e_score_correction_bias", model.num_experts, 4)
            shared = _require_divisible(model.num_shared_experts * model.moe_intermediate_size, tp, "shared intermediate")
            linear(mlp + ".shared_experts.gate_up_proj", 2 * shared, h)
            linear(mlp + ".shared_experts.down_proj", h, shared)
        else:
            dense = _require_divisible(model.intermediate_size, tp, "dense intermediate")
            linear(mlp + ".gate_up_proj", 2 * dense, h)
            linear(mlp + ".down_proj", h, dense)
    return entries


def glm_block_fp8_postload_parameter_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    hardware_id: str,
    linear_backend: str,
    routed_backend: str,
    include_mla_absorption: bool,
) -> dict[str, int]:
    """Registered payload under an explicit SM90 DeepGEMM source profile.

    vLLM's CUDA block processing leaves E4M3 weights/FP32 scales in their
    registered shapes. DeepGEMM's pinned SM90 (128,128) SFB branch checks and
    returns the same scale tensor, including its FP32 dtype; it neither
    broadcasts scales to full rows nor packs them as SM100 INT UE8M0. Optional
    value requantization is in-place. MoE uses that same conversion routine.
    Parameter replacement and ordinary MLA views follow the loader inventory.
    This does not count kernel state/workspaces, peak temporaries, driver or
    allocator reserves, and does not establish actual provider selection.
    """
    if (hardware_id != "nvidia_h200" or linear_backend != "deep_gemm"
            or routed_backend != "deep_gemm"):
        raise NotImplementedError("GLM post-load payload requires the explicit H200 SM90 DeepGEMM linear/MoE source profile")
    # The owner validates exact config, block geometry, TP/EP and dtype here.
    # No storage delta is introduced by this source branch; do not charge
    # both old and replacement Parameter wrappers for their shared payload.
    return glm_block_fp8_loader_payload_bytes(
        model, parallelism, include_mla_absorption=include_mla_absorption)


def glm_block_fp8_deepgemm_workspace_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    hardware_id: str,
    tokens: int,
) -> dict[str, int]:
    """One declared GLM-5.3 DeepGEMM MoE workspace request, not total peak.

    CUDA/SM90, EP-off, one unchunked call, BF16 activations, no CPU expert
    counts, no alignment override, one workspace lane/ubatch. The pinned
    SM90 alignment is 128. TritonOrDeepGemmExperts.workspace_shapes selects
    DeepGemmExperts for these aligned shapes even when apply later chooses
    Triton. The source bounds routed rows by M*topk plus padding for at most
    min(M*topk, local_experts) active experts, then rounds to 128.

    FusedMoEKernel._allocate_buffers initially views output in workspace13; its
    two simultaneous requests share one WorkspaceManager allocation, each
    view rounded to 256 bytes. Layers reuse this owner, so do not sum the
    same request over layers. Other requests may grow that allocation;
    apply allocates a separate output and replaces the initial output view;
    that output, quantization, dispatch and library scratch are excluded here.
    No runtime/backend selection or normal-capacity assertion is implied.
    """
    padded, intermediate, k = _glm_block_fp8_deepgemm_call_geometry(
        model, parallelism, hardware_id=hardware_id, tokens=tokens)
    n = 2 * intermediate
    workspace13_elements = padded * max(intermediate, k)
    workspace2_elements = padded * max(n, k)
    output_elements = tokens * k
    # The initial output view shares the first request. apply's standalone
    # output is outside this shared allocation and inventoried separately.
    common_bytes = 2 * max(workspace13_elements, output_elements)
    second_bytes = 2 * workspace2_elements
    total = -(-common_bytes // 256) * 256 + -(-second_bytes // 256) * 256
    return {"worker.shared_moe_workspace": total}


def _glm_block_fp8_deepgemm_call_geometry(
    model: ModelConfig, parallelism: ParallelConfig, *, hardware_id: str, tokens: int,
) -> tuple[int, int, int]:
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("GLM DeepGEMM workspace tokens must be a positive integer")
    # Reuse the exact source-config, dtype, TP/EP and hardware boundary.
    glm_block_fp8_postload_parameter_bytes(
        model, parallelism, hardware_id=hardware_id,
        linear_backend="deep_gemm", routed_backend="deep_gemm",
        include_mla_absorption=False)
    if parallelism.enable_expert_parallel:
        raise NotImplementedError("GLM DeepGEMM workspace requires EP-off; received-token geometry is unbound")
    alignment = 128
    intermediate = _require_divisible(
        model.moe_intermediate_size, parallelism.tensor_parallel_size,
        "routed intermediate")
    n, k = 2 * intermediate, model.hidden_size
    if tokens < alignment or n % alignment or k % alignment:
        raise NotImplementedError("GLM DeepGEMM workspace requires the aligned M>=128 source branch; Triton workspace is unbound")
    routed = tokens * model.num_experts_per_tok
    padded = routed + min(routed, model.num_experts) * (alignment - 1)
    padded = -(-padded // alignment) * alignment
    return padded, intermediate, k


def glm_block_fp8_deepgemm_call_allocation_bytes(
    model: ModelConfig, parallelism: ParallelConfig, *, hardware_id: str, tokens: int,
) -> dict[str, int]:
    """Unique tensor allocations outside the shared workspace for one call.

    Same explicit SM90/EP-off/unchunked BF16/SILU FP8-block source scenario
    as glm_block_fp8_deepgemm_workspace_bytes: no expert metadata, no LoRA,
    router weight on output, ordinary FP32 (not UE8M0) scales. Modular apply
    allocates output independently and uses that as the final output alias.
    Prepare quantizes input; deepgemm_moe_permute allocates scales and INT32
    metadata; fused SiLU/mul/quant allocates only its FP32 scales because its
    FP8 output is a workspace13 view. All MM results also reuse workspace.

    Each value is one object's payload, not an allocation count times layers.
    The sum is a conservative call allocation envelope, not a simultaneous
    live peak: permutation-local count/start buffers can die before activation
    scales exist. Caller inputs/routing outputs, shared experts, library state
    or scratch, allocator overhead and whole-worker reserves are excluded.
    No device/provider execution or normal-capacity acceptance is implied.
    """
    padded, intermediate, k = _glm_block_fp8_deepgemm_call_geometry(
        model, parallelism, hardware_id=hardware_id, tokens=tokens)
    return {
        "moe.call.output": tokens * k * 2,
        "moe.prepare.quantized_input": tokens * k,
        "moe.prepare.input_scales": tokens * (k // 128) * 4,
        "moe.permute.expert_start_loc": model.num_experts * 4,
        "moe.permute.input_scales": padded * (k // 128) * 4,
        "moe.permute.expert_ids": padded * 4,
        "moe.permute.inv_perm": tokens * model.num_experts_per_tok * 4,
        "moe.permute.expert_num_tokens": model.num_experts * 4,
        "moe.activation.output_scales": padded * (intermediate // 128) * 4,
    }


def glm_block_fp8_constructor_buffer_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    max_batched_tokens: int,
    max_requests: int,
) -> dict[str, int]:
    """Persistent source tensor payload outside registered GLM-5.3 parameters.

    Ordinary CUDA SparseMLACommonImpl/CP1/PP1, target-off, eager and no HiSparse; model construction
    runs in the loader's BF16 default-dtype context. Counts one shared logical
    top-k object, separate physical/count/request objects per index group, and
    one shared default-RoPE cache. Streams/events, runtime allocator overhead,
    backend workspaces, post-load conversions and KV remain outside this scope.
    This is a constructor scenario, not observed residency or available-KV.
    Keys name unique logical object owners, not state_dict parameter paths.
    """
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("GLM constructor buffers require ModelConfig and ParallelConfig")
    if (model.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S
            or model.compute_dtype != "bf16" or model.target_semantics is None):
        raise NotImplementedError("GLM constructor buffers require the exact BF16-compute GLM block-FP8 config")
    if not parallelism.is_rank_local or parallelism.pipeline_parallel_size != 1:
        raise NotImplementedError("GLM constructor buffers require PP=DP=CP=SP=1")
    for value in (max_batched_tokens, max_requests):
        if type(value) is not int or value <= 0:
            raise ValueError("GLM constructor buffer limits must be explicit positive integers")
    if max_requests > max_batched_tokens:
        raise ValueError("GLM constructor request limit cannot exceed the native batch-token limit")
    semantics = model.target_semantics
    if (semantics.rope_type != "default" or semantics.rope_layout != "interleaved"
            or semantics.indexer_rope_layout != semantics.rope_layout):
        raise NotImplementedError("GLM constructor buffers require identical default main/indexer RoPE cache keys")
    entries = {"model.topk_indices_buffer": max_batched_tokens * model.index_topk * 4}
    # get_sparse_mla_index_group_max_rows: MTP-off max_query_len=1.
    workspace_rows = min(max_batched_tokens, max_requests)
    current_group = -1
    for indexer_type in model.indexer_types:
        if indexer_type not in ("full", "shared"):
            raise ValueError("GLM constructor buffers require full/shared indexer types")
        # SparseMLAIndexGroupBuilder also creates a group for an initial
        # consumer without an index-producing predecessor.
        if indexer_type == "full" or current_group == -1:
            current_group += 1
            prefix = f"model.index_groups.{current_group}"
            entries[prefix + ".physical_topk_indices"] = (workspace_rows + 1) * model.index_topk * 4
            entries[prefix + ".valid_topk_counts"] = (workspace_rows + 1) * 4
            entries[prefix + ".request_ids"] = workspace_rows * 4
    # get_rope caches by dimension, max_position, style, parameters and dtype.
    # All layers and both main/indexer calls use the same key here. The cache
    # spans the HF maximum, not the shorter Scheduler max_model_len input.
    entries["model.shared_rotary.cos_sin_cache"] = model.max_position_embeddings * model.qk_rope_head_dim * 2
    return entries


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
        # Indexer.k_norm uses vLLM LayerNorm, whose constructor explicitly
        # registers both affine parameters in FP32, independent of activations.
        total += 2 * model.index_head_dim * 4
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
    *,
    share_target_vocabulary: bool = False,
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
    if share_target_vocabulary:
        # vLLM 0.30 llm_base_proposer and GPU-v2 eagle/utils both replace
        # every MTP shared_head.head with the target head. Embedding sharing
        # is restricted to PP1; PP>1 retains the draft's own embedding.
        draft_head_bytes = 0
        if parallelism.pipeline_parallel_size == 1:
            draft_embedding_bytes = 0
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
    model_execution_version: str | None = None,
    v4_weight_storage_profile: str | None = None,
) -> tuple[ModelRankResidency, ...]:
    """Return source-defined text-model weight bytes for every pool rank.

    A ModelConfig extracted from a multimodal text_config carries only the
    text submodel. This diagnostic does not account for the parent tower or
    projector; normal capacity must reject that incomplete inventory unless
    a complete outer-model residency/skip/loader binding is supported.

    Explicit vllm_0_30_0 supports dense BF16 Qwen3 TP1/PP1 target-off and
    the BF16 GLM source scenario with
    ordinary vocab-parallel embeddings (VLLM_REPLICATE_EMBED disabled).
    It accounts for source-defined post-load MTP vocabulary aliases, not
    actual loader execution, reserves or native PP/runtime acceptance.
    None preserves the existing independent draft-vocabulary inventory.
    V4 additionally requires an explicit SM90 DeepGEMM/Marlin source storage
    profile; it includes the root mHC parameters separately from decoder
    layers. Model buffers and runtime reserves are owned by the memory
    policy, not counted as weights here. The profile does not select or
    verify a runtime provider. Explicit ``ffn`` additionally exposes the
    existing matched-TP dense BF16 AF post-attention norm/MLP inventory;
    embedding, final norm and vocabulary head remain on Attention. This
    weight-only role does not establish a KV plan or complete worker budget.
    """

    if not isinstance(model, ModelConfig):
        raise TypeError("model residency requires ModelConfig")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("model residency requires ParallelConfig")
    if not isinstance(include_mtp, bool):
        raise TypeError("model residency include_mtp must be boolean")
    if v4_weight_storage_profile is not None:
        if (v4_weight_storage_profile != "sm90_deepgemm_marlin" or model.model_type != "deepseek_v4"
                or model_execution_version != "vllm_0_30_0" or worker_role != "colocated" or include_mtp):
            raise NotImplementedError("V4 weight storage profile requires explicit0.30 colocated target-off SM90 DeepGEMM/Marlin")
        payload = dsv4_postload_parameter_bytes(model, parallelism, hardware_id="nvidia_h200",
            linear_backend="deep_gemm", routed_backend="marlin")
        embedding = payload["model.embed_tokens.weight"]
        head = payload["lm_head.weight"]
        norm = payload["model.norm.weight"]
        decoder = sum(size for path, size in payload.items() if path.startswith("model.layers."))
        additional = sum(payload.values()) - embedding - head - norm - decoder
        return tuple(ModelRankResidency(rank, rank, 0, 0, 0, model.num_hidden_layers,
            embedding, decoder, 0, norm, head, additional) for rank in range(parallelism.tensor_parallel_size))
    if (model.model_type == "glm_moe_dsa" and model.weight_dtype == "fp8"
            and model.quantized_weight_targets):
        raise NotImplementedError(
            "official GLM block-FP8 normal residency requires a bound backend, "
            "complete persistent buffers, kernel state, workspaces and reserves; "
            "scoped parameter payloads are insufficient")
    if model_execution_version not in (None, "vllm_0_30_0"):
        raise ValueError("unsupported model residency execution version")
    whole_model_roles = {"colocated", "prefill", "decode"}
    attention_roles = {"attention", "prefill-attention", "decode-attention"}
    ffn_role = worker_role == "ffn"
    if ffn_role and (model.model_type not in ("qwen3", "llama") or model.num_experts
            or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
            or model.hidden_act != "silu" or model.mlp_bias or model.rms_norm_eps is None
            or include_mtp or model.num_mtp_layers or parallelism.tensor_parallel_size not in (1, 2)
            or parallelism.pipeline_parallel_size != 1 or parallelism.enable_expert_parallel
            or not parallelism.is_rank_local):
        raise NotImplementedError("AF FFN residency requires dense BF16 Qwen3/Llama matched TP1/2 PP1 EP/SP/CP/DP/MTP-off")
    if model_execution_version == "vllm_0_30_0":
        if model.model_type in ("qwen3", "llama"):
            # Qwen3Model inherits Qwen2Model's embedding/decoder/norm
            # placement. Qwen3ForCausalLM ties its padded ParallelLMHead to
            # embed_tokens when configured. The shared dense inventory below
            # includes Q/K RMSNorm weights and counts this alias once.
            if (model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
                    or model.num_experts or (model.model_type == "qwen3" and not model.qk_norm) or include_mtp
                    or model.num_mtp_layers or worker_role not in whole_model_roles | attention_roles | {"ffn"}
                    or (worker_role in attention_roles and parallelism.tensor_parallel_size not in (
                        (1, 2) if worker_role == "attention" else (1,)))
                    or (parallelism.pipeline_parallel_size != 1 and not (
                        worker_role == "colocated"
                        and is_validated_qwen3_pipeline_model_scope(model, parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size)))
                    or not parallelism.is_rank_local or parallelism.enable_expert_parallel):
                raise NotImplementedError("vLLM0.30 dense Qwen3/Llama residency requires BF16 PP1/EP-off/MTP-off whole-model pools")
        elif model.model_type == "qwen3_moe":
            # 0.30 Qwen3Moe retains the shared padded vocabulary/GQA
            # inventory. EP owns complete experts; _moe_parameter_count
            # partitions E by EP rather than dividing their I by TP again.
            if (not is_validated_qwen3_moe_ep2_model_scope(model)
                    or include_mtp or worker_role not in whole_model_roles | attention_roles
                    or (worker_role in attention_roles and (parallelism.tensor_parallel_size != 1
                        or parallelism.enable_expert_parallel))
                    or parallelism.pipeline_parallel_size != 1
                    or parallelism.tensor_parallel_size not in (1, 2)
                    or not parallelism.is_rank_local
                    or (parallelism.enable_expert_parallel
                        and parallelism.tensor_parallel_size != 2)):
                raise NotImplementedError(
                    "vLLM0.30 Qwen3MoE residency requires validated BF16 "
                    "TP1/2 PP1/MTP-off whole-model pools, with optional TP=EP=2")
        elif (model.model_type != "glm_moe_dsa" or model.compute_dtype != "bf16"
                or model.weight_dtype != "bf16" or model.tie_word_embeddings):
            raise NotImplementedError(
                "vLLM 0.30 residency requires BF16 GLM with separate target vocabulary weights or dense BF16 Qwen3")
    if worker_role not in whole_model_roles | attention_roles | {"ffn"}:
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
        mtp_bytes = _target_mtp_weight_bytes(
            model, parallelism,
            share_target_vocabulary=model_execution_version == "vllm_0_30_0")
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
        elif ffn_role:
            # The shared decoder inventory is exactly input-norm+attention
            # plus RemoteFFN's post-attention norm and MLP. Reuse that source
            # partition rather than copying a second parameter formula.
            decoder_parameters = sum(_decoder_layer_parameter_count(model, parallelism, layer_index)
                - _attention_layer_parameter_count(model, parallelism)
                for layer_index in range(layer_start, layer_end))
            decoder_bytes = decoder_parameters * element_bytes
        else:
            raise AssertionError("validated worker role disappeared")
        for tp_rank in range(tp_size):
            rank = pp_rank * tp_size + tp_rank
            embedding_bytes = (
                embedding_parameters * element_bytes if pp_rank == 0 and not ffn_role else 0)
            final_norm_bytes = (
                model.hidden_size * element_bytes
                if pp_rank == pp_size - 1 and not ffn_role else 0)
            lm_head_bytes = (
                embedding_parameters * element_bytes
                if pp_rank == pp_size - 1
                and not model.tie_word_embeddings and not ffn_role else 0)
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
