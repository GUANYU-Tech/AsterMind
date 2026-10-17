"""Validated LLM model configuration used by Serving workload construction."""

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.serving.workload.rope_dispatch import VLLM_ROPE_CLASS_BY_SCALING_TYPE


_MODEL_SCHEMAS = {
    "qwen3": {
        "architecture": "Qwen3ForCausalLM",
        "attention_layout": "standard_gqa",
        "head_dim": "required",
        "mlp_bias": False,
        "moe": False,
        "expert_count_field": None,
        "dense_layer_policy": "all",
        "mtp_architecture": "none",
        "qk_norm": True,
    },
    "qwen3_moe": {
        "architecture": "Qwen3MoeForCausalLM",
        "attention_layout": "standard_gqa",
        "head_dim": "required",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "num_experts",
        "dense_layer_policy": "none",
        "mtp_architecture": "none",
        "qk_norm": True,
    },
    "llama": {
        "architecture": "LlamaForCausalLM",
        "attention_layout": "standard_gqa",
        "head_dim": "derived",
        "mlp_bias": False,
        "moe": False,
        "expert_count_field": None,
        "dense_layer_policy": "all",
        "mtp_architecture": "none",
        "qk_norm": False,
    },
    "glm4_moe": {
        "architecture": "Glm4MoeForCausalLM",
        "attention_layout": "standard_gqa",
        "head_dim": "required",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "n_routed_experts",
        "dense_layer_policy": "first_k_dense_replace",
        "mtp_architecture": "glm4_moe_mtp",
        "qk_norm": "config",
    },
    "glm_moe_dsa": {
        "architecture": "GlmMoeDsaForCausalLM",
        "attention_layout": "dsa_mla",
        "head_dim": "mla",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "n_routed_experts",
        "dense_layer_policy": "first_k_dense_replace",
        "mtp_architecture": "glm_moe_dsa_mtp",
        "qk_norm": False,
    },
    "kimi_k2": {
        "architecture": "DeepseekV3ForCausalLM",
        "attention_layout": "mla",
        "head_dim": "mla",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "n_routed_experts",
        "dense_layer_policy": "first_k_dense_replace",
        "mtp_architecture": "none_zero_allowed",
        "qk_norm": False,
    },
    "deepseek_v3": {
        "architecture": "DeepseekV3ForCausalLM",
        "attention_layout": "mla",
        "head_dim": "mla",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "n_routed_experts",
        "dense_layer_policy": "first_k_dense_replace",
        "mtp_architecture": "deepseek_mtp",
        "qk_norm": False,
    },
}


_COMPUTE_DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
    "uint8": 1,
}

_CONFIG_DTYPE_NAMES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float8_e4m3fn": "fp8",
    "int8": "int8",
    "uint8": "uint8",
}

_CONFIG_CANONICALIZATION = "json_sort_keys_compact_utf8_v1"

# On-disk config.json identities of the pinned fixed-target model files.
# ``_repository_target_provenance`` keys on exactly these digests; the
# Scheduler, workload-construction, and emission gates reference the same
# identities through these constants instead of restating the hashes.
KIMI_K2_CONFIG_SHA256 = (
    "03af24788bd7788c65db1fcb8326c5f5c5270aab717c442df67ba9db01041b3d")
GLM_MOE_DSA_CONFIG_SHA256 = (
    "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a")
GLM_MOE_DSA_FP8_CONFIG_SHA256 = (
    "fb2555ddea2a16f6aa9915a6d7a5acca1cd7dd8406255e66b65ab48805ca4f3d")
DEEPSEEK_V3_CONFIG_SHA256 = (
    "cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9")
# DeepSeek-V3 with only num_hidden_layers reduced to 32, the local
# deployment variant that fits TP4 on four H200s.  Its source is the
# derived-local config itself, so its source and on-disk digests coincide.
DEEPSEEK_V3_L32_CONFIG_SHA256 = (
    "d0d3a3a2583fc419d18dd314c928524733aaaf227ad045f592c5e8e9b09375d1")

# Every pinned target config identity across the three fixed target models.
# Gates that accept "any named target file" (the vLLM Scheduler scope) use
# this flat set; the per-model pairing lives in
# TARGET_MODEL_CONFIG_SHA256_BY_TYPE.
TARGET_MODEL_CONFIG_SHA256 = frozenset((
    KIMI_K2_CONFIG_SHA256,
    GLM_MOE_DSA_CONFIG_SHA256,
    GLM_MOE_DSA_FP8_CONFIG_SHA256,
    DEEPSEEK_V3_CONFIG_SHA256,
    DEEPSEEK_V3_L32_CONFIG_SHA256,
))

# The pinned config identities accepted for each fixed target model:
# Kimi-K2.6 ships one file, GLM-5.2 ships the BF16 and FP8 files, and
# DeepSeek-V3 ships the official config plus the 32-layer local variant.
TARGET_MODEL_CONFIG_SHA256_BY_TYPE = {
    "kimi_k2": (KIMI_K2_CONFIG_SHA256,),
    "glm_moe_dsa": (
        GLM_MOE_DSA_CONFIG_SHA256,
        GLM_MOE_DSA_FP8_CONFIG_SHA256,
    ),
    "deepseek_v3": (
        DEEPSEEK_V3_CONFIG_SHA256,
        DEEPSEEK_V3_L32_CONFIG_SHA256,
    ),
}


def is_named_target_model_identity(model_type: str, config_sha256: str) -> bool:
    """Return whether the pair names one of the pinned target model files.

    The check is deliberately flat across models -- it matches the
    historical Scheduler gate, which accepts any pinned config identity with
    any target model_type; the per-model pairing is enforced separately by
    the workload identity contract.
    """

    return (
        model_type in TARGET_MODEL_CONFIG_SHA256_BY_TYPE
        and config_sha256 in TARGET_MODEL_CONFIG_SHA256)


_TARGET_MODEL_PROVENANCE = {
    "glm_moe_dsa": {
        "model_id": "zai-org/GLM-5.2",
        "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
        "source_config_sha256": (
            "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"
        ),
        "canonical_config_sha256": (
            "ea00f6b92243d88fd4ed8b921438cecf0923bd27d10a1a4626aa570bf94000d0"
        ),
        "source_url": (
            "https://huggingface.co/zai-org/GLM-5.2/resolve/"
            "b4734de4facf877f85769a911abafc5283eab3d9/config.json"
        ),
        "extraction_path": "$",
        "extracted_config_sha256": (
            "ea00f6b92243d88fd4ed8b921438cecf0923bd27d10a1a4626aa570bf94000d0"
        ),
    },
    "kimi_k2": {
        "model_id": "moonshotai/Kimi-K2.6",
        "revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
        "source_config_sha256": (
            "85825ca6e18cbe539eb83ee09eedfb3f4222265929f06e9f535a6d9364f55899"
        ),
        "canonical_config_sha256": (
            "d0df2342b7d2fbc4bc3b4e5d9b190d2de7b0df063365d2910eb48599d67aa4a9"
        ),
        "source_url": (
            "https://huggingface.co/moonshotai/Kimi-K2.6/resolve/"
            "7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json"
        ),
        "extraction_path": "text_config",
        "extracted_config_sha256": (
            "a2b365e4927bc86375829321d37bb3741b4d8d5f48aa243a9ba550740b5d66fd"
        ),
    },
    "deepseek_v3": {
        "model_id": "deepseek-ai/DeepSeek-V3",
        "revision": "e815299b0bcbac849fa540c768ef21845365c9eb",
        "source_config_sha256": (
            "cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9"
        ),
        "canonical_config_sha256": (
            "fed8c13b4637058cd68e600bd4bf7dc734bda4594dd583e3b49fa27c6e123cc6"
        ),
        "source_url": (
            "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/"
            "e815299b0bcbac849fa540c768ef21845365c9eb/config.json"
        ),
        "extraction_path": "$",
        "extracted_config_sha256": (
            "fed8c13b4637058cd68e600bd4bf7dc734bda4594dd583e3b49fa27c6e123cc6"
        ),
    },
}

_GLM_FP8_PROVENANCE = {
    "model_id": "zai-org/GLM-5.2",
    "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
    "source_config_sha256": "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a",
    "canonical_config_sha256": "2b89e4e8383bc9ff7c557516b9180d2173188db9d8baeb4e2b3f3caa1666023c",
    "source_url": "derived-local://glm-5.2-fp8?base_config_sha256=185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a&injected=quantization_config",
    "extraction_path": "$",
    "extracted_config_sha256": "2b89e4e8383bc9ff7c557516b9180d2173188db9d8baeb4e2b3f3caa1666023c",
}

# DeepSeek-V3 with only num_hidden_layers reduced to 32, the local
# deployment variant that fits TP4 on four H200s with dummy FP8 weights so
# a real vLLM run and the simulator see the identical architecture.
_DSV3_L32_PROVENANCE = {
    "model_id": "deepseek-ai/DeepSeek-V3",
    "revision": "e815299b0bcbac849fa540c768ef21845365c9eb",
    "source_config_sha256": "d0d3a3a2583fc419d18dd314c928524733aaaf227ad045f592c5e8e9b09375d1",
    "canonical_config_sha256": "6d2217b99dac0684b5d3c0ffca5855d39a802112f091b7267924a1b92b8067be",
    "source_url": "derived-local://deepseek-v3-l32?base_config_sha256=cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9&override=num_hidden_layers:32",
    "extraction_path": "$",
    "extracted_config_sha256": "6d2217b99dac0684b5d3c0ffca5855d39a802112f091b7267924a1b92b8067be",
}


@dataclass(frozen=True)
class TargetModelSemantics:
    """Source-pinned semantics that distinguish the two target models."""

    workload_contract: str
    use_cache: bool
    moe_layer_frequency: int
    moe_router_dtype: str
    moe_score_function: str
    moe_topk_method: str
    moe_group_count: int
    moe_topk_group_count: int
    moe_normalize_topk_probability: bool
    rope_type: str
    rope_theta: float
    rope_layout: str
    rope_factor: float | None
    rope_original_max_position_embeddings: int | None
    rope_beta_fast: float | None
    rope_beta_slow: float | None
    rope_mscale: float | None
    rope_mscale_all_dim: float | None
    index_skip_topk_offset: int | None
    index_topk_frequency: int | None
    index_topk_pattern: tuple[str, ...] | None
    indexer_rope_layout: str | None
    mtp_runtime_status: str

    def __post_init__(self) -> None:
        for field in (
                "workload_contract", "moe_router_dtype", "moe_score_function",
                "moe_topk_method", "rope_type", "rope_layout",
                "mtp_runtime_status"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"TargetModelSemantics.{field} must be non-empty")
        if not isinstance(self.use_cache, bool):
            raise TypeError("TargetModelSemantics.use_cache must be boolean")
        if not isinstance(self.moe_normalize_topk_probability, bool):
            raise TypeError(
                "TargetModelSemantics.moe_normalize_topk_probability must "
                "be boolean")
        for field in (
                "moe_layer_frequency", "moe_group_count",
                "moe_topk_group_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"TargetModelSemantics.{field} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"TargetModelSemantics.{field} must be positive")
        if self.moe_topk_group_count > self.moe_group_count:
            raise ValueError(
                "target MoE top-k group count cannot exceed group count")
        if self.moe_score_function != "sigmoid":
            raise ValueError("target MoE score function must be sigmoid")
        if self.moe_router_dtype not in ("bf16", "fp32"):
            raise ValueError(
                "target MoE router dtype must be bf16 or fp32")
        if self.moe_topk_method != "noaux_tc":
            raise ValueError("target MoE top-k method must be noaux_tc")
        if self.rope_type not in ("default", "deepseek_yarn"):
            raise ValueError(
                f"unsupported target RoPE type: {self.rope_type!r}")
        if self.rope_layout != "interleaved":
            raise ValueError("target main RoPE layout must be interleaved")
        if (isinstance(self.rope_theta, bool)
                or not isinstance(self.rope_theta, (int, float))
                or not math.isfinite(self.rope_theta)
                or self.rope_theta <= 0):
            raise ValueError("target RoPE theta must be finite and positive")
        yarn_fields = (
            self.rope_factor,
            self.rope_original_max_position_embeddings,
            self.rope_beta_fast,
            self.rope_beta_slow,
            self.rope_mscale,
            self.rope_mscale_all_dim,
        )
        if self.rope_type == "default":
            if any(value is not None for value in yarn_fields):
                raise ValueError("default target RoPE cannot carry YaRN fields")
        elif self.rope_type == "deepseek_yarn":
            if any(value is None for value in yarn_fields):
                raise ValueError("DeepSeek YaRN requires every scaling field")
            numeric_yarn_fields = (
                self.rope_factor,
                self.rope_beta_fast,
                self.rope_beta_slow,
                self.rope_mscale,
                self.rope_mscale_all_dim,
            )
            if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in numeric_yarn_fields):
                raise ValueError(
                    "DeepSeek YaRN numeric fields must be finite and positive")
            original_max = self.rope_original_max_position_embeddings
            if (isinstance(original_max, bool)
                    or not isinstance(original_max, int)
                    or original_max <= 0):
                raise ValueError(
                    "DeepSeek YaRN original max position must be positive")
        else:
            raise AssertionError(
                f"unhandled target RoPE type: {self.rope_type!r}")
        dsa_fields = (
            self.index_skip_topk_offset,
            self.index_topk_frequency,
            self.indexer_rope_layout,
        )
        if all(value is None for value in dsa_fields):
            if self.index_topk_pattern is not None:
                raise ValueError(
                    "plain MLA cannot carry an index top-k pattern")
        elif any(value is None for value in dsa_fields):
            raise ValueError(
                "DSA target semantics require all indexer control fields")
        else:
            for field in ("index_skip_topk_offset", "index_topk_frequency"):
                value = getattr(self, field)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"TargetModelSemantics.{field} must be an integer")
                if value <= 0:
                    raise ValueError(
                        f"TargetModelSemantics.{field} must be positive")
            if self.indexer_rope_layout != "interleaved":
                raise ValueError("DSA indexer RoPE layout must be interleaved")
            if (self.index_topk_pattern is not None
                    and any(value not in ("F", "S")
                            for value in self.index_topk_pattern)):
                raise ValueError(
                    "DSA index top-k pattern entries must be F or S")


def expand_models_root(model_path: str) -> str:
    """Resolve the legacy ``${MODELS_ROOT}`` hardware-execution placeholder."""
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be a non-empty string")
    if "${MODELS_ROOT}" in model_path:
        if "MODELS_ROOT" not in os.environ:
            raise RuntimeError(
                "MODELS_ROOT must be set to resolve model paths containing "
                f"${{MODELS_ROOT}}: {model_path}")
        return model_path.replace("${MODELS_ROOT}", os.environ["MODELS_ROOT"])
    return model_path


def _load_model_json(model_path: str) -> tuple[
        dict[str, object], str, str, str, str | None, str | None, str | None,
        str]:
    path = Path(model_path)
    config_path = path if path.is_file() else path / "config.json"
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    cfg = json.loads(config_bytes)
    if not isinstance(cfg, dict):
        raise TypeError("model config root must be a mapping")
    canonical_config_bytes = json.dumps(
        cfg,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical_config_sha256 = hashlib.sha256(
        canonical_config_bytes).hexdigest()
    provenance_path = path / "model_provenance.json" if path.is_dir() else None
    if provenance_path is None or not provenance_path.exists():
        repository_target = _repository_target_provenance(config_sha256)
        if repository_target is None:
            if (cfg["model_type"] == "glm_moe_dsa"
                    and "quantization_config" in cfg):
                raise ValueError(
                    "GLM FP8 model config is not registered in the repository "
                    f"provenance registry: {config_sha256}")
            return (
                cfg, config_sha256, config_sha256, canonical_config_sha256,
                None, None, None, "text")
        if repository_target is not None:
            return (
                cfg,
                config_sha256,
                repository_target["source_config_sha256"],
                canonical_config_sha256,
                repository_target["model_id"],
                repository_target["revision"],
                repository_target["source_url"],
                "text",
            )
        return (
            cfg,
            config_sha256,
            config_sha256,
            canonical_config_sha256,
            None,
            None,
            None,
            "text",
        )
    raw_provenance = json.loads(provenance_path.read_bytes())
    if not isinstance(raw_provenance, dict):
        raise TypeError("model provenance root must be a mapping")
    required = {
        "schema_version",
        "model_id",
        "revision",
        "source_config_sha256",
        "fixture_config_sha256",
        "source_url",
        "model_scope",
    }
    if set(raw_provenance) != required:
        raise ValueError(
            "model provenance fields must be exactly "
            f"{sorted(required)}, got {sorted(raw_provenance)}")
    if raw_provenance["schema_version"] != 1:
        raise ValueError("model provenance schema_version must be 1")
    for field in required - {"schema_version"}:
        value = raw_provenance[field]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"model provenance {field} must be a non-empty string")
    fixture_sha256 = raw_provenance["fixture_config_sha256"]
    if fixture_sha256 != config_sha256:
        raise ValueError(
            "model provenance fixture hash disagrees with config.json: "
            f"expected={fixture_sha256}, actual={config_sha256}")
    source_sha256 = raw_provenance["source_config_sha256"]
    for label, digest in (
            ("source_config_sha256", source_sha256),
            ("fixture_config_sha256", fixture_sha256)):
        if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                f"model provenance {label} must be a lowercase SHA256")
    model_scope = raw_provenance["model_scope"]
    if model_scope != "text":
        raise ValueError(
            f"unsupported model provenance scope: {model_scope!r}")
    return (
        cfg,
        config_sha256,
        source_sha256,
        canonical_config_sha256,
        raw_provenance["model_id"],
        raw_provenance["revision"],
        raw_provenance["source_url"],
        model_scope,
    )


def _repository_target_provenance(
    config_sha256: str,
) -> dict[str, str] | None:
    target_by_repository_sha = {
        GLM_MOE_DSA_CONFIG_SHA256: (
            "glm_moe_dsa"
        ),
        KIMI_K2_CONFIG_SHA256: (
            "kimi_k2"
        ),
        GLM_MOE_DSA_FP8_CONFIG_SHA256: (
            "glm_moe_dsa_fp8"
        ),
        DEEPSEEK_V3_CONFIG_SHA256: (
            "deepseek_v3"
        ),
        DEEPSEEK_V3_L32_CONFIG_SHA256: (
            "deepseek_v3_l32"
        ),
    }
    if config_sha256 not in target_by_repository_sha:
        return None
    target = target_by_repository_sha[config_sha256]
    if target == "glm_moe_dsa_fp8":
        return _GLM_FP8_PROVENANCE
    if target == "deepseek_v3_l32":
        return _DSV3_L32_PROVENANCE
    return _TARGET_MODEL_PROVENANCE[target]


def _text_model_config(
    root: dict[str, object],
    *,
    has_fixed_provenance: bool,
) -> tuple[dict[str, object], str]:
    model_type = root["model_type"]
    if model_type == "kimi_k25":
        if not has_fixed_provenance:
            raise ValueError(
                "Kimi-K2.6 text extraction requires fixed model provenance")
        architectures = root["architectures"]
        if architectures != ["KimiK25ForConditionalGeneration"]:
            raise ValueError(
                "kimi_k25 requires KimiK25ForConditionalGeneration")
        text_config = root["text_config"]
        if not isinstance(text_config, dict):
            raise TypeError("kimi_k25 text_config must be a mapping")
        if root["dtype"] != text_config["dtype"]:
            raise ValueError(
                "kimi_k25 container dtype disagrees with text_config")
        return text_config, "text_config"
    elif "text_config" in root:
        raise ValueError(
            f"unsupported nested text model container: {model_type!r}")
    else:
        return root, "$"


@dataclass(frozen=True)
class ModelConfig:
    """LLM architecture parameters needed for kernel dimension inference."""
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int
    vocab_size: int
    # Canonical compute dtype.  Element width is derived from this field;
    # callers must never infer dtype semantics from byte width.
    compute_dtype: str
    weight_dtype: str
    weight_group_size: int
    quantized_weight_targets: tuple[str, ...]
    quantized_weight_ignore: tuple[str, ...]
    attention_layout: str
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: tuple[str, ...]
    index_share_for_mtp_iteration: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    # Architecture name from HF config, retained as explicit model provenance.
    # Dense/MoE behavior is determined by validated expert fields, not by
    # substring matching on this display name.
    architecture: str
    # Whether token embedding and LM-head weights share checkpoint storage.
    # Pipeline reference coverage currently requires this to be false.
    tie_word_embeddings: bool
    # Whether linears carry a bias. Both values are required from the model
    # config because they participate in the measured launch-data key.
    attention_bias: bool
    mlp_bias: bool
    # Rotary position table size (HF config max_position_embeddings).
    max_position_embeddings: int
    model_type: str
    num_dense_layers: int
    num_shared_experts: int
    num_mtp_layers: int
    mtp_architecture: str
    qk_norm: bool
    # HuggingFace ``rope_scaling``/``rope_parameters`` scaling type verbatim
    # ("default" when the config carries no scaling block).  vLLM's
    # ``get_rope`` turns this string into one concrete rotary class, and the
    # out-of-tree custom-op registry is keyed on that class name, so this field
    # -- not the architecture -- decides which physical rope program runs.
    # See src/serving/workload/rope_dispatch.py.
    rope_scaling_type: str
    routed_scaling_factor: float
    target_semantics: TargetModelSemantics | None
    config_sha256: str
    source_config_sha256: str
    canonical_config_sha256: str
    config_canonicalization: str
    config_extraction_path: str
    extracted_config_sha256: str
    model_id: str | None
    model_revision: str | None
    model_source_url: str | None
    model_scope: str

    def __post_init__(self) -> None:
        positive = (
            "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "num_hidden_layers", "head_dim",
            "vocab_size", "max_position_embeddings",
        )
        for field in positive:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ModelConfig.{field} must be an integer")
            if value <= 0:
                raise ValueError(f"ModelConfig.{field} must be positive")
        if not isinstance(self.attention_layout, str):
            raise TypeError("ModelConfig.attention_layout must be a string")
        if self.attention_layout not in (
                "standard_gqa", "mla", "dsa_mla"):
            raise ValueError(
                f"unsupported attention layout: {self.attention_layout!r}")
        if not isinstance(self.model_type, str) or not self.model_type:
            raise ValueError("ModelConfig.model_type must be non-empty")
        if self.model_type not in _MODEL_SCHEMAS:
            raise ValueError(
                f"unsupported ModelConfig.model_type: {self.model_type!r}")
        if (not isinstance(self.rope_scaling_type, str)
                or not self.rope_scaling_type):
            raise ValueError(
                "ModelConfig.rope_scaling_type must be a non-empty string")
        if self.rope_scaling_type not in VLLM_ROPE_CLASS_BY_SCALING_TYPE:
            raise ValueError(
                "unsupported ModelConfig.rope_scaling_type: "
                f"{self.rope_scaling_type!r}")
        attention_fields = (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
        )
        for field in attention_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ModelConfig.{field} must be an integer")
            if value < 0:
                raise ValueError(f"ModelConfig.{field} cannot be negative")
        if not isinstance(self.indexer_types, tuple) or any(
                not isinstance(value, str) or not value
                for value in self.indexer_types):
            raise TypeError(
                "ModelConfig.indexer_types must be a tuple of non-empty "
                "strings")
        if not isinstance(self.index_share_for_mtp_iteration, bool):
            raise TypeError(
                "ModelConfig.index_share_for_mtp_iteration must be boolean")
        mla_fields = (
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        if self.attention_layout == "standard_gqa":
            head_dim_policy = _MODEL_SCHEMAS[self.model_type]["head_dim"]
            if head_dim_policy == "derived":
                if self.hidden_size % self.num_attention_heads:
                    raise ValueError(
                        "ModelConfig.hidden_size must be divisible by "
                        "num_attention_heads for a derived head_dim")
                if self.head_dim != (
                        self.hidden_size // self.num_attention_heads):
                    raise ValueError(
                        "ModelConfig.head_dim must equal hidden_size divided "
                        "by num_attention_heads for a derived head_dim")
            elif head_dim_policy == "required":
                pass
            else:
                raise ValueError(
                    "standard GQA requires an explicit or derived head_dim "
                    f"policy, got {head_dim_policy!r}")
            if self.num_attention_heads % self.num_key_value_heads:
                raise ValueError(
                    "ModelConfig.num_attention_heads must be divisible by "
                    "num_key_value_heads")
            if any(mla_fields) or any((
                    self.index_n_heads,
                    self.index_head_dim,
                    self.index_topk,
            )) or self.indexer_types or self.index_share_for_mtp_iteration:
                raise ValueError(
                    "standard GQA cannot carry MLA or DSA parameters")
        elif self.attention_layout in ("mla", "dsa_mla"):
            if any(value <= 0 for value in mla_fields):
                raise ValueError("MLA requires positive projection dimensions")
            if self.head_dim != (
                    self.qk_nope_head_dim + self.qk_rope_head_dim):
                raise ValueError(
                    "MLA head_dim must equal qk_nope_head_dim plus "
                    "qk_rope_head_dim")
            if self.attention_layout == "mla":
                if any((
                        self.index_n_heads,
                        self.index_head_dim,
                        self.index_topk,
                )) or self.indexer_types or self.index_share_for_mtp_iteration:
                    raise ValueError("plain MLA cannot carry DSA parameters")
            elif self.attention_layout == "dsa_mla":
                if min(
                        self.index_n_heads,
                        self.index_head_dim,
                        self.index_topk) <= 0:
                    raise ValueError("DSA requires positive indexer dimensions")
                if len(self.indexer_types) != self.num_hidden_layers:
                    raise ValueError(
                        "DSA indexer_types must cover every hidden layer")
                if any(value not in ("full", "shared")
                       for value in self.indexer_types):
                    raise ValueError(
                        "DSA indexer type must be 'full' or 'shared'")
            else:
                raise AssertionError(
                    f"unhandled attention layout: {self.attention_layout!r}")
        else:
            raise AssertionError(
                f"unhandled attention layout: {self.attention_layout!r}")
        if not isinstance(self.architecture, str):
            raise TypeError("ModelConfig.architecture must be a string")
        if not self.architecture.strip():
            raise ValueError("ModelConfig.architecture must be non-empty")
        if not isinstance(self.compute_dtype, str):
            raise TypeError("ModelConfig.compute_dtype must be a string")
        if self.compute_dtype not in _COMPUTE_DTYPE_BYTES:
            raise ValueError(
                f"unsupported compute dtype: {self.compute_dtype!r}")
        if self.weight_dtype not in (
                set(_COMPUTE_DTYPE_BYTES) | {"mixed_int4_bf16"}):
            raise ValueError(
                f"unsupported weight dtype: {self.weight_dtype!r}")
        if (isinstance(self.weight_group_size, bool)
                or not isinstance(self.weight_group_size, int)):
            raise TypeError("ModelConfig.weight_group_size must be an integer")
        for field in ("quantized_weight_targets", "quantized_weight_ignore"):
            value = getattr(self, field)
            if not isinstance(value, tuple) or any(
                    not isinstance(item, str) or not item for item in value):
                raise TypeError(
                    f"ModelConfig.{field} must be a tuple of non-empty "
                    "strings")
        if self.weight_dtype == "mixed_int4_bf16":
            if self.compute_dtype != "bf16":
                raise ValueError(
                    "mixed INT4/BF16 weights require BF16 compute dtype")
            if self.weight_group_size <= 0:
                raise ValueError(
                    "mixed INT4/BF16 weights require a positive group size")
            if self.quantized_weight_targets != ("Linear",):
                raise ValueError(
                    "mixed INT4/BF16 weights currently require target Linear")
            if not self.quantized_weight_ignore:
                raise ValueError(
                    "mixed INT4/BF16 weights require explicit ignore rules")
        elif self.weight_dtype in _COMPUTE_DTYPE_BYTES:
            if (self.weight_dtype != self.compute_dtype
                    and not (
                        self.model_type in ("glm_moe_dsa", "deepseek_v3")
                        and self.compute_dtype == "bf16"
                        and self.weight_dtype == "fp8")):
                raise ValueError(
                    "unquantized weight dtype must equal compute dtype")
            if self.weight_group_size != 0:
                raise ValueError(
                    "unquantized weights require weight_group_size=0")
            if self.quantized_weight_targets or self.quantized_weight_ignore:
                raise ValueError(
                    "unquantized weights cannot carry quantization selectors")
        else:
            raise AssertionError(
                f"unhandled weight dtype: {self.weight_dtype!r}")
        if self.model_scope != "text":
            raise ValueError(
                f"unsupported ModelConfig.model_scope: {self.model_scope!r}")
        for field in (
                "config_sha256",
                "source_config_sha256",
                "canonical_config_sha256",
                "extracted_config_sha256"):
            digest = getattr(self, field)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    f"ModelConfig.{field} must be a lowercase SHA256")
        if self.config_extraction_path not in ("$", "text_config"):
            raise ValueError(
                "ModelConfig.config_extraction_path must be '$' or "
                "'text_config'")
        if self.config_canonicalization != _CONFIG_CANONICALIZATION:
            raise ValueError(
                "unsupported model config canonicalization: "
                f"{self.config_canonicalization!r}")
        provenance_values = (
            self.model_id,
            self.model_revision,
            self.model_source_url,
        )
        if any(value is None for value in provenance_values):
            if not all(value is None for value in provenance_values):
                raise ValueError(
                    "model ID, revision, and source URL must be all present "
                    "or all null")
        elif any(not isinstance(value, str) or not value
                 for value in provenance_values):
            raise ValueError(
                "model ID, revision, and source URL must be non-empty")
        for field in (
                "num_dense_layers", "num_shared_experts", "num_mtp_layers"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ModelConfig.{field} must be an integer")
            if value < 0:
                raise ValueError(f"ModelConfig.{field} cannot be negative")
        if self.num_dense_layers > self.num_hidden_layers:
            raise ValueError(
                "ModelConfig.num_dense_layers cannot exceed hidden layers")
        if not isinstance(self.mtp_architecture, str):
            raise TypeError("ModelConfig.mtp_architecture must be a string")
        if self.mtp_architecture not in (
                "none", "glm4_moe_mtp", "glm_moe_dsa_mtp", "deepseek_mtp"):
            raise ValueError(
                f"unsupported MTP architecture: {self.mtp_architecture!r}")
        if self.num_mtp_layers == 0:
            if self.mtp_architecture != "none":
                raise ValueError(
                    "zero MTP layers require mtp_architecture='none'")
        elif self.mtp_architecture == "none":
            raise ValueError(
                "positive MTP layers require an MTP architecture")
        for field in (
                "tie_word_embeddings", "attention_bias", "mlp_bias",
                "qk_norm"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"ModelConfig.{field} must be boolean")
        if (isinstance(self.routed_scaling_factor, bool)
                or not isinstance(self.routed_scaling_factor, (int, float))
                or not math.isfinite(self.routed_scaling_factor)
                or self.routed_scaling_factor <= 0):
            raise ValueError(
                "ModelConfig.routed_scaling_factor must be finite and "
                "positive")
        if self.target_semantics is not None and not isinstance(
                self.target_semantics, TargetModelSemantics):
            raise TypeError(
                "ModelConfig.target_semantics must be TargetModelSemantics "
                "or null")
        if self.model_type in ("glm_moe_dsa", "kimi_k2", "deepseek_v3"):
            if self.target_semantics is None:
                raise ValueError(
                    "fixed target models require source-pinned semantics")
        elif self.target_semantics is not None:
            raise ValueError(
                "non-target models cannot carry target-model semantics")
        for field in (
                "num_experts", "num_experts_per_tok",
                "moe_intermediate_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ModelConfig.{field} must be an integer")
        moe_values = (
            self.num_experts,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
        )
        if all(value == 0 for value in moe_values):
            if self.num_dense_layers != self.num_hidden_layers:
                raise ValueError("dense models require every layer to be dense")
            if self.num_shared_experts != 0:
                raise ValueError("dense models cannot contain shared experts")
            return
        if any(value <= 0 for value in moe_values):
            raise ValueError(
                "MoE models require positive expert count, top-k, and "
                "moe_intermediate_size")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        if self.num_dense_layers == self.num_hidden_layers:
            raise ValueError("MoE models require at least one routed MoE layer")

    @property
    def dtype_bytes(self) -> int:
        """Element width derived from the canonical compute dtype."""

        return _COMPUTE_DTYPE_BYTES[self.compute_dtype]

    def weight_dtype_for_module(
        self,
        module_path: str,
        target_class: str,
    ) -> str:
        """Resolve the fixed checkpoint's storage dtype for one module."""

        for label, value in (
                ("module_path", module_path),
                ("target_class", target_class)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if self.weight_dtype != "mixed_int4_bf16":
            return self.weight_dtype
        if target_class not in self.quantized_weight_targets:
            return self.compute_dtype
        for raw_pattern in self.quantized_weight_ignore:
            if not raw_pattern.startswith("re:"):
                raise ValueError(
                    "quantized weight ignore rules must use the re: prefix")
            if re.fullmatch(raw_pattern[3:], module_path):
                return self.compute_dtype
        return "int4"

    def kv_heads_per_tensor_parallel_rank(self, tp_size: int) -> int:
        """Return the standard-GQA KV heads resident on each TP rank."""

        if isinstance(tp_size, bool) or not isinstance(tp_size, int):
            raise TypeError("tensor parallel size must be an integer")
        if tp_size < 1:
            raise ValueError("tensor parallel size must be positive")
        if self.attention_layout != "standard_gqa":
            raise NotImplementedError(
                "MLA/DSA KV capacity requires a compressed-KV and index-cache "
                "storage contract")
        kv_heads = self.num_key_value_heads
        if kv_heads >= tp_size and kv_heads % tp_size == 0:
            return kv_heads // tp_size
        if kv_heads < tp_size and tp_size % kv_heads == 0:
            return 1
        raise ValueError(
            "num_key_value_heads and tensor parallel size must divide one "
            f"another exactly: {kv_heads} vs {tp_size}")

    def kv_bytes_per_token_per_tensor_parallel_rank(
        self,
        tp_size: int,
    ) -> int:
        """Return resident K+V bytes per token on one standard-GQA TP rank."""

        return self.kv_bytes_per_token_per_parallel_rank(tp_size, 1)

    def _standard_gqa_kv_bytes(
            self, local_layers: int, tensor_parallel_size: int) -> int:
        """Return this rank's key+value bytes for a standard-GQA layout.

        A GQA layer keeps one key and one value head per local KV head, so the
        per-rank cache is two copies of the local head count times head width,
        scaled by the rank's layer count and the storage dtype.  Both the
        whole-plan and the layer-range callers compute the same quantity from
        different layer counts.
        """

        local_kv_heads = self.kv_heads_per_tensor_parallel_rank(
            tensor_parallel_size)
        return (
            2 * local_layers * local_kv_heads
            * self.head_dim * self.dtype_bytes)


    def kv_bytes_per_token_per_parallel_rank(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ) -> int:
        """Return resident attention-cache bytes on one parallel rank.

        Standard GQA stores one K and one V tensor.  MLA stores the single
        compressed ``kv_c || k_pe`` vector.  Pinned vLLM 0.27.1 GLM-5.2
        semantics add a separate FP8 index cache on full-indexer layers.  The
        DSA total is exact for PP=1; PP>1 remains unsupported because each
        stage owns a different number of full-indexer layers.
        """

        if (isinstance(tensor_parallel_size, bool)
                or not isinstance(tensor_parallel_size, int)):
            raise TypeError("tensor parallel size must be an integer")
        if tensor_parallel_size < 1:
            raise ValueError("tensor parallel size must be positive")
        if (isinstance(pipeline_parallel_size, bool)
                or not isinstance(pipeline_parallel_size, int)):
            raise TypeError("pipeline parallel size must be an integer")
        if pipeline_parallel_size < 1:
            raise ValueError("pipeline parallel size must be positive")
        if self.num_hidden_layers % pipeline_parallel_size:
            raise ValueError(
                "num_hidden_layers must be divisible by pipeline parallel "
                "size: "
                f"{self.num_hidden_layers} vs {pipeline_parallel_size}")
        local_layers = self.num_hidden_layers // pipeline_parallel_size
        if self.attention_layout == "standard_gqa":
            return self._standard_gqa_kv_bytes(
                local_layers, tensor_parallel_size)
        if self.attention_layout == "dsa_mla":
            if pipeline_parallel_size != 1:
                raise NotImplementedError(
                    "DSA KV capacity for PP>1 requires rank-specific full-"
                    "indexer layer placement")
            if self.num_attention_heads % tensor_parallel_size:
                raise ValueError(
                    "DSA query heads must be divisible by tensor parallel "
                    f"size: {self.num_attention_heads} vs "
                    f"{tensor_parallel_size}")
            compressed_bytes = (
                self.num_hidden_layers
                * (self.kv_lora_rank + self.qk_rope_head_dim)
                * self.dtype_bytes
            )
            index_scale_bytes = self.index_head_dim // 128 * 4
            full_indexer_layers = self.indexer_types.count("full")
            index_bytes = full_indexer_layers * (
                self.index_head_dim + index_scale_bytes)
            return compressed_bytes + index_bytes
        if self.attention_layout != "mla":
            raise AssertionError(
                f"unhandled attention layout: {self.attention_layout!r}")
        if self.num_attention_heads % tensor_parallel_size:
            raise ValueError(
                "MLA query heads must be divisible by tensor parallel size: "
                f"{self.num_attention_heads} vs {tensor_parallel_size}")

        return (
            local_layers
            * (self.kv_lora_rank + self.qk_rope_head_dim)
            * self.dtype_bytes
        )

    def kv_bytes_per_token_for_layer_range(
        self,
        tensor_parallel_size: int,
        layer_start: int,
        layer_end: int,
    ) -> int:
        """Return exact rank-local KV bytes for a contiguous PP layer range."""

        for name, value in (
                ("tensor_parallel_size", tensor_parallel_size),
                ("layer_start", layer_start), ("layer_end", layer_end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if not 0 <= layer_start < layer_end <= self.num_hidden_layers:
            raise ValueError("KV layer range must be non-empty and in bounds")
        local_layers = layer_end - layer_start
        if self.attention_layout == "standard_gqa":
            return self._standard_gqa_kv_bytes(
                local_layers, tensor_parallel_size)
        if self.num_attention_heads % tensor_parallel_size:
            raise ValueError(
                "MLA/DSA query heads must be divisible by tensor parallel "
                f"size: {self.num_attention_heads} vs {tensor_parallel_size}")
        compressed_bytes = (
            local_layers
            * (self.kv_lora_rank + self.qk_rope_head_dim)
            * self.dtype_bytes)
        if self.attention_layout == "mla":
            return compressed_bytes
        if self.attention_layout == "dsa_mla":
            index_scale_bytes = self.index_head_dim // 128 * 4
            full_indexer_layers = self.indexer_types[
                layer_start:layer_end].count("full")
            return compressed_bytes + full_indexer_layers * (
                self.index_head_dim + index_scale_bytes)
        raise AssertionError(
            f"unhandled attention layout: {self.attention_layout!r}")

    def layer_uses_moe(self, layer_index: int) -> bool:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer")
        if not 0 <= layer_index < self.num_hidden_layers:
            raise ValueError("layer_index is outside the target model")
        if self.target_semantics is None:
            return bool(
                self.num_experts and layer_index >= self.num_dense_layers)
        return bool(
            self.num_experts
            and layer_index >= self.num_dense_layers
            and layer_index % self.target_semantics.moe_layer_frequency == 0)

    def moe_layer_ordinal(self, layer_index: int) -> "int | None":
        """Position of a MoE layer among the model's MoE layers, else None."""

        if not self.layer_uses_moe(layer_index):
            return None
        return sum(
            self.layer_uses_moe(index) for index in range(layer_index))

    def mtp_layer_uses_moe(self, mtp_layer_index: int) -> bool:
        if (isinstance(mtp_layer_index, bool)
                or not isinstance(mtp_layer_index, int)):
            raise TypeError("mtp_layer_index must be an integer")
        if not 0 <= mtp_layer_index < self.num_mtp_layers:
            raise ValueError("mtp_layer_index is outside the MTP model")
        if self.mtp_architecture == "glm4_moe_mtp":
            # vLLM 0.11 overrides the draft config to num_hidden_layers=0;
            # Glm4MoeMultiTokenPredictor therefore cycles weights starting at
            # decoder layer index zero.
            return bool(self.num_experts and 0 >= self.num_dense_layers)
        elif self.mtp_architecture == "glm_moe_dsa_mtp":
            if self.target_semantics is None:
                raise ValueError(
                    "GLM-5.2 MTP layer mapping requires target semantics")
            layer_index = self.num_hidden_layers + mtp_layer_index
            return bool(
                self.num_experts
                and layer_index >= self.num_dense_layers
                and layer_index
                % self.target_semantics.moe_layer_frequency == 0)
        elif self.mtp_architecture == "deepseek_mtp":
            if self.target_semantics is None:
                raise ValueError(
                    "DeepSeek-V3 MTP layer mapping requires target semantics")
            # vLLM DeepSeekMTP appends the NextN layer directly after the
            # main decoder stack, so it lands past first_k_dense_replace and
            # is a routed MoE layer under moe_layer_freq=1.
            layer_index = self.num_hidden_layers + mtp_layer_index
            return bool(
                self.num_experts
                and layer_index >= self.num_dense_layers
                and layer_index
                % self.target_semantics.moe_layer_frequency == 0)
        elif self.mtp_architecture == "none":
            raise ValueError("a model without MTP has no MTP decoder layers")
        else:
            raise ValueError(
                f"unsupported MTP architecture: {self.mtp_architecture!r}")


def is_validated_llama_tp2_model_scope(model: ModelConfig) -> bool:
    """Recognize the exact Llama config with source-backed TP=2 coverage."""

    if not isinstance(model, ModelConfig):
        raise TypeError("Llama TP=2 scope requires a ModelConfig")
    return (
        model.config_sha256
        == "ad0330f06f0e3d4defab454b7bfdff2ed3e89d2a6e7ae4679483e0cd1bfe3512"
        and model.model_type == "llama"
        and model.architecture == "LlamaForCausalLM"
        and model.hidden_size == 128
        and model.intermediate_size == 256
        and model.num_attention_heads == 2
        and model.num_key_value_heads == 1
        and model.num_hidden_layers == 1
        and model.head_dim == 64
        and model.vocab_size == 256
        and model.max_position_embeddings == 128
        and model.attention_layout == "standard_gqa"
        and not model.qk_norm
        and model.num_experts == 0
        and model.num_mtp_layers == 0
        and not model.tie_word_embeddings
        and not model.attention_bias
        and not model.mlp_bias
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
    )


def is_validated_qwen3_moe_ep2_model_scope(model: ModelConfig) -> bool:
    """Return whether Qwen3MoE has the source-backed TP=EP=2 semantics."""

    if not isinstance(model, ModelConfig):
        raise TypeError("Qwen3MoE TP=EP=2 scope requires a ModelConfig")
    return (
        model.architecture == "Qwen3MoeForCausalLM"
        and model.model_type == "qwen3_moe"
        and model.attention_layout == "standard_gqa"
        and model.num_experts > 0
        and model.num_experts % 2 == 0
        and 0 < model.num_experts_per_tok <= model.num_experts
        and model.num_dense_layers == 0
        and model.num_shared_experts == 0
        and model.num_mtp_layers == 0
        and model.mtp_architecture == "none"
        and model.moe_intermediate_size % 2 == 0
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
        and model.weight_group_size == 0
        and model.quantized_weight_targets == ()
        and model.quantized_weight_ignore == ()
    )


def is_validated_qwen3_tp2_host_kv_model_scope(
    model: ModelConfig,
) -> bool:
    """Recognize the exact Qwen3 config used by the TP=2 Host-KV gate."""

    if not isinstance(model, ModelConfig):
        raise TypeError("Qwen3 TP=2 Host-KV scope requires a ModelConfig")
    return (
        model.config_sha256
        == "0c128be62e30095587ed663284f6e7f09b4473909dd1b1428f3babc798f14c90"
        and model.model_type == "qwen3"
        and model.architecture == "Qwen3ForCausalLM"
        and model.hidden_size == 256
        and model.intermediate_size == 512
        and model.num_attention_heads == 4
        and model.num_key_value_heads == 2
        and model.num_hidden_layers == 2
        and model.head_dim == 64
        and model.vocab_size == 64
        and model.max_position_embeddings == 128
        and model.attention_layout == "standard_gqa"
        and model.qk_norm
        and model.num_experts == 0
        and model.num_mtp_layers == 0
        and not model.tie_word_embeddings
        and not model.attention_bias
        and not model.mlp_bias
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
    )


def _rope_scaling_type(cfg: dict) -> str:
    """Read the HuggingFace rope scaling type that selects the vLLM class.

    ``get_rope`` reads ``rope_parameters["rope_type"]`` in the newer configs
    and the legacy ``rope_scaling`` block otherwise, where the key is spelled
    ``rope_type`` (llama-3.x, qwen3) or ``type`` (DeepSeek-family).  Absent or
    null scaling means the ``"default"`` branch.
    """

    present = tuple(
        name for name in ("rope_parameters", "rope_scaling") if name in cfg)
    if not present:
        # get_rope reads rope_parameters.get("rope_type", "default"), so a
        # config without any scaling block takes the "default" branch.
        return "default"
    if len(present) != 1:
        raise ValueError(
            "model config carries both rope_parameters and rope_scaling")
    raw = cfg[present[0]]
    if raw is None:
        return "default"
    if not isinstance(raw, dict):
        raise TypeError(f"model config {present[0]} must be a mapping or null")
    if "mrope_section" in raw:
        raise ValueError(
            "multimodal RoPE (mrope_section) is not a supported Serving "
            "rope program")
    if "use_fope" in raw and raw["use_fope"]:
        raise ValueError(
            "Fourier RoPE (use_fope) is not a supported Serving rope program")
    if "rope_type" in raw:
        scaling_type = raw["rope_type"]
    elif "type" in raw:
        scaling_type = raw["type"]
    else:
        raise ValueError(
            f"model config {present[0]} carries no rope type field")
    if not isinstance(scaling_type, str) or not scaling_type:
        raise ValueError(
            f"model config {present[0]} rope type must be a non-empty string")
    if scaling_type not in VLLM_ROPE_CLASS_BY_SCALING_TYPE:
        raise ValueError(
            f"unsupported vLLM rope scaling type: {scaling_type!r}")
    return scaling_type


def load_model_config(model_path: str | Path) -> ModelConfig:
    """Load a repository JSON file or a HuggingFace model directory."""
    model_path = expand_models_root(str(model_path))
    (
        root_cfg,
        config_sha256,
        source_config_sha256,
        canonical_config_sha256,
        model_id,
        model_revision,
        model_source_url,
        model_scope,
    ) = _load_model_json(model_path)
    cfg, config_extraction_path = _text_model_config(
        root_cfg,
        has_fixed_provenance=model_id is not None,
    )
    extracted_config_sha256 = hashlib.sha256(json.dumps(
        cfg,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

    hidden_size = cfg["hidden_size"]
    num_attention_heads = cfg["num_attention_heads"]
    num_key_value_heads = cfg["num_key_value_heads"]
    model_type = cfg["model_type"]
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("model config model_type must be a non-empty string")
    if model_type not in _MODEL_SCHEMAS:
        raise ValueError(f"unsupported model_type: {model_type!r}")
    if model_type in ("glm_moe_dsa", "kimi_k2", "deepseek_v3") and model_id is None:
        raise ValueError(
            f"model_type {model_type!r} requires fixed model provenance")
    if model_type in _TARGET_MODEL_PROVENANCE:
        if config_sha256 == GLM_MOE_DSA_FP8_CONFIG_SHA256:
            expected_provenance = _GLM_FP8_PROVENANCE
        elif config_sha256 == DEEPSEEK_V3_L32_CONFIG_SHA256:
            expected_provenance = _DSV3_L32_PROVENANCE
        else:
            expected_provenance = _TARGET_MODEL_PROVENANCE[model_type]
        actual_provenance = {
            "model_id": model_id,
            "revision": model_revision,
            "source_config_sha256": source_config_sha256,
            "canonical_config_sha256": canonical_config_sha256,
            "source_url": model_source_url,
            "extraction_path": config_extraction_path,
            "extracted_config_sha256": extracted_config_sha256,
        }
        if actual_provenance != expected_provenance:
            raise ValueError(
                f"model_type {model_type!r} provenance disagrees with the "
                f"fixed target revision: actual={actual_provenance}, "
                f"expected={expected_provenance}")
    schema = _MODEL_SCHEMAS[model_type]
    architectures = cfg["architectures"]
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("model config requires exactly one architecture")
    architecture = architectures[0]
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("model config architecture must be a non-empty string")
    if architecture != schema["architecture"]:
        raise ValueError(
            f"model_type {model_type!r} requires architecture "
            f"{schema['architecture']!r}, got {architecture!r}")

    expert_fields = {
        "num_experts", "num_local_experts", "num_experts_per_tok",
        "experts_per_token", "moe_intermediate_size", "n_routed_experts",
        "n_shared_experts",
    }
    present_expert_fields = expert_fields & set(cfg)
    if schema["moe"] is True:
        expert_count_field = schema["expert_count_field"]
        if not isinstance(expert_count_field, str):
            raise AssertionError(
                f"MoE schema lacks expert count for {model_type!r}")
        required_expert_fields = {
            expert_count_field, "num_experts_per_tok", "moe_intermediate_size"}
        if model_type in ("glm4_moe", "glm_moe_dsa", "kimi_k2", "deepseek_v3"):
            required_expert_fields.add("n_shared_experts")
        if present_expert_fields != required_expert_fields:
            raise ValueError(
                f"model_type {model_type!r} requires expert fields "
                f"{sorted(required_expert_fields)}, got "
                f"{sorted(present_expert_fields)}")
        num_experts = cfg[expert_count_field]
        num_experts_per_tok = cfg["num_experts_per_tok"]
        moe_intermediate_size = cfg["moe_intermediate_size"]
        if model_type in ("glm4_moe", "glm_moe_dsa", "kimi_k2", "deepseek_v3"):
            shared_experts = cfg["n_shared_experts"]
            if (isinstance(shared_experts, bool)
                    or not isinstance(shared_experts, int)
                    or shared_experts <= 0):
                raise ValueError(
                    f"model_type {model_type!r} requires a positive "
                    "n_shared_experts value")
            num_shared_experts = shared_experts
        elif model_type == "qwen3_moe":
            num_shared_experts = 0
        else:
            raise AssertionError(
                f"unexpected MoE model_type: {model_type!r}")
    elif schema["moe"] is False:
        if present_expert_fields:
            raise ValueError(
                f"dense model_type {model_type!r} contains expert fields: "
                f"{sorted(present_expert_fields)}")
        num_experts = 0
        num_experts_per_tok = 0
        moe_intermediate_size = 0
        num_shared_experts = 0
    else:
        raise AssertionError(f"invalid model schema for {model_type!r}")

    if (isinstance(hidden_size, bool) or not isinstance(hidden_size, int)
            or isinstance(num_attention_heads, bool)
            or not isinstance(num_attention_heads, int)
            or isinstance(num_key_value_heads, bool)
            or not isinstance(num_key_value_heads, int)):
        raise TypeError(
            "model hidden_size, num_attention_heads, and "
            "num_key_value_heads must be integers")
    if (hidden_size <= 0 or num_attention_heads <= 0
            or num_key_value_heads <= 0):
        raise ValueError(
            "model hidden_size, num_attention_heads, and "
            "num_key_value_heads must be positive")
    if num_attention_heads % num_key_value_heads:
        raise ValueError(
            "model num_attention_heads must be divisible by "
            "num_key_value_heads")
    if schema["head_dim"] == "required":
        head_dim = cfg["head_dim"]
        if (isinstance(head_dim, bool) or not isinstance(head_dim, int)):
            raise TypeError("model head_dim must be an integer")
        if head_dim <= 0:
            raise ValueError("model head_dim must be positive")
    elif schema["head_dim"] == "derived":
        if hidden_size % num_attention_heads:
            raise ValueError(
                "model hidden_size must be divisible by num_attention_heads "
                "when head_dim is derived")
        derived_head_dim = hidden_size // num_attention_heads
        head_dim = derived_head_dim
        if "head_dim" in cfg and cfg["head_dim"] != derived_head_dim:
            raise ValueError(
                f"model head_dim {cfg['head_dim']!r} disagrees with derived "
                f"value {derived_head_dim}")
    elif schema["head_dim"] == "mla":
        mla_dimension_fields = (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
        )
        for field in mla_dimension_fields:
            value = cfg[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"model config {field} must be an integer")
            if value <= 0:
                raise ValueError(f"model config {field} must be positive")
        head_dim = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
        if "qk_head_dim" in cfg and cfg["qk_head_dim"] != head_dim:
            raise ValueError(
                "model qk_head_dim disagrees with qk_nope_head_dim plus "
                "qk_rope_head_dim")
        if ("head_dim" in cfg
                and cfg["head_dim"] != cfg["qk_nope_head_dim"]):
            raise ValueError(
                "MLA config head_dim must equal qk_nope_head_dim")
    else:
        raise AssertionError(f"invalid head_dim schema for {model_type!r}")

    attention_layout = schema["attention_layout"]
    if attention_layout == "standard_gqa":
        q_lora_rank = 0
        kv_lora_rank = 0
        qk_nope_head_dim = 0
        qk_rope_head_dim = 0
        v_head_dim = 0
        index_n_heads = 0
        index_head_dim = 0
        index_topk = 0
        indexer_types = ()
        index_share_for_mtp_iteration = False
    elif attention_layout in ("mla", "dsa_mla"):
        q_lora_rank = cfg["q_lora_rank"]
        kv_lora_rank = cfg["kv_lora_rank"]
        qk_nope_head_dim = cfg["qk_nope_head_dim"]
        qk_rope_head_dim = cfg["qk_rope_head_dim"]
        v_head_dim = cfg["v_head_dim"]
        if attention_layout == "mla":
            index_n_heads = 0
            index_head_dim = 0
            index_topk = 0
            indexer_types = ()
            index_share_for_mtp_iteration = False
        elif attention_layout == "dsa_mla":
            index_n_heads = cfg["index_n_heads"]
            index_head_dim = cfg["index_head_dim"]
            index_topk = cfg["index_topk"]
            raw_indexer_types = cfg["indexer_types"]
            if not isinstance(raw_indexer_types, list):
                raise TypeError("DSA indexer_types must be a list")
            indexer_types = tuple(raw_indexer_types)
            index_share_for_mtp_iteration = cfg[
                "index_share_for_mtp_iteration"]
        else:
            raise AssertionError(
                f"unhandled attention layout: {attention_layout!r}")
    else:
        raise AssertionError(
            f"invalid attention layout schema for {model_type!r}")

    dtype_fields = tuple(
        field for field in ("dtype", "torch_dtype") if field in cfg)
    if not dtype_fields:
        raise ValueError("model config requires dtype or torch_dtype")
    compute_dtypes = []
    for field in dtype_fields:
        raw_dtype = cfg[field]
        if not isinstance(raw_dtype, str):
            raise TypeError(f"model config {field} must be a string")
        dtype_name = raw_dtype.lower()
        if dtype_name not in _CONFIG_DTYPE_NAMES:
            raise ValueError(
                f"unsupported model {field}: {dtype_name!r}")
        compute_dtypes.append(_CONFIG_DTYPE_NAMES[dtype_name])
    if len(set(compute_dtypes)) != 1:
        raise ValueError(
            "model config dtype and torch_dtype disagree: "
            f"{compute_dtypes}")
    compute_dtype = compute_dtypes[0]
    if "quantization_config" in cfg:
        quantization = cfg["quantization_config"]
        if model_type == "glm_moe_dsa":
            if (not isinstance(quantization, dict)
                    or quantization["quant_method"] != "fp8"
                    or quantization["weight_dtype"] != "fp8"
                    or quantization["compute_dtype"] != "bfloat16"
                    or quantization["scope"] != "linear"):
                raise ValueError("GLM FP8 requires its explicit linear-weight schema")
            weight_dtype = "fp8"
            weight_group_size = 0
            quantized_weight_targets = ()
            quantized_weight_ignore = ()
        elif model_type == "deepseek_v3":
            if not isinstance(quantization, dict):
                raise TypeError("model quantization_config must be a mapping")
            if set(quantization) != {
                    "activation_scheme", "fmt", "quant_method",
                    "weight_block_size"}:
                raise ValueError(
                    "DeepSeek-V3 requires the official FP8 quantization "
                    "schema fields")
            if (quantization["quant_method"] != "fp8"
                    or quantization["fmt"] != "e4m3"
                    or quantization["activation_scheme"] != "dynamic"
                    or quantization["weight_block_size"] != [128, 128]):
                raise ValueError(
                    "DeepSeek-V3 requires the official block-wise 128x128 "
                    "e4m3 FP8 schema with dynamic activations")
            # The official DeepSeek-V3 release stores weights as e4m3 with
            # 128x128 block scales.  The Serving FP8 contract models weight
            # storage as one byte per element and has no scale-granularity
            # dimension anywhere downstream (FUSED_MOE requires
            # weight_group_size=0 for fp8; residency counts element bytes),
            # so block-wise and per-tensor scaling are the same model here.
            weight_dtype = "fp8"
            weight_group_size = 0
            quantized_weight_targets = ()
            quantized_weight_ignore = ()
        else:
            if model_type != "kimi_k2":
                raise ValueError(
                    f"model_type {model_type!r} does not support a quantization "
                    "config in the Serving model contract")
            if not isinstance(quantization, dict):
                raise TypeError("model quantization_config must be a mapping")
            if (quantization["quant_method"] != "compressed-tensors"
                    or quantization["quantization_status"] != "compressed"
                    or quantization["format"] != "pack-quantized"):
                raise ValueError(
                    "Kimi-K2.6 requires its fixed compressed-tensors weight format")
            config_groups = quantization["config_groups"]
            if not isinstance(config_groups, dict) or set(config_groups) != {"group_0"}:
                raise ValueError("Kimi-K2.6 quantization requires exactly config group_0")
            group = config_groups["group_0"]
            if not isinstance(group, dict):
                raise TypeError("quantization group_0 must be a mapping")
            raw_targets = group["targets"]
            if (not isinstance(raw_targets, list)
                    or any(not isinstance(value, str) or not value for value in raw_targets)):
                raise TypeError("quantization targets must be a list of non-empty strings")
            weights = group["weights"]
            if not isinstance(weights, dict):
                raise TypeError("quantization group weights must be a mapping")
            if (weights["type"] != "int" or weights["num_bits"] != 4
                    or weights["strategy"] != "group" or weights["symmetric"] is not True):
                raise ValueError("Kimi-K2.6 requires symmetric groupwise INT4 weights")
            weight_group_size = weights["group_size"]
            if (isinstance(weight_group_size, bool)
                    or not isinstance(weight_group_size, int) or weight_group_size <= 0):
                raise ValueError("Kimi-K2.6 INT4 weight group size must be positive")
            raw_ignore = quantization["ignore"]
            if (not isinstance(raw_ignore, list)
                    or any(not isinstance(value, str) or not value for value in raw_ignore)):
                raise TypeError("quantization ignore rules must be a list of non-empty strings")
            weight_dtype = "mixed_int4_bf16"
            quantized_weight_targets = tuple(raw_targets)
            quantized_weight_ignore = tuple(raw_ignore)
    else:
        weight_dtype = compute_dtype
        weight_group_size = 0
        quantized_weight_targets = ()
        quantized_weight_ignore = ()

    attention_bias_field = cfg["attention_bias"]
    if not isinstance(attention_bias_field, bool):
        raise ValueError("model config attention_bias must be boolean")
    if "tie_word_embeddings" not in cfg:
        raise ValueError(
            "model config requires tie_word_embeddings")
    tie_word_embeddings = cfg["tie_word_embeddings"]
    if not isinstance(tie_word_embeddings, bool):
        raise ValueError(
            "model config tie_word_embeddings must be boolean")
    mlp_bias_field = schema["mlp_bias"]
    if "mlp_bias" in cfg and cfg["mlp_bias"] != mlp_bias_field:
        raise ValueError(
            f"model config mlp_bias {cfg['mlp_bias']!r} disagrees with "
            f"model_type {model_type!r} schema {mlp_bias_field!r}")

    dense_layer_policy = schema["dense_layer_policy"]
    if model_type == "qwen3_moe":
        if ("decoder_sparse_step" not in cfg
                or cfg["decoder_sparse_step"] != 1):
            raise ValueError(
                "Qwen3MoE requires decoder_sparse_step=1 for the canonical "
                "all-MoE layer contract")
        if ("mlp_only_layers" not in cfg
                or cfg["mlp_only_layers"] != []):
            raise ValueError(
                "Qwen3MoE requires empty mlp_only_layers for the canonical "
                "all-MoE layer contract")
    if dense_layer_policy == "all":
        num_dense_layers = cfg["num_hidden_layers"]
    elif dense_layer_policy == "none":
        num_dense_layers = 0
    elif dense_layer_policy == "first_k_dense_replace":
        num_dense_layers = cfg["first_k_dense_replace"]
    else:
        raise AssertionError(
            f"invalid dense layer policy for {model_type!r}")
    if model_type == "glm_moe_dsa":
        raw_mlp_layer_types = cfg["mlp_layer_types"]
        if not isinstance(raw_mlp_layer_types, list):
            raise TypeError("GLM-5.2 mlp_layer_types must be a list")
        expected_mlp_layer_types = (
            ["dense"] * num_dense_layers
            + ["sparse"] * (
                cfg["num_hidden_layers"] - num_dense_layers)
        )
        if raw_mlp_layer_types != expected_mlp_layer_types:
            raise ValueError(
                "GLM-5.2 mlp_layer_types disagree with the dense/MoE "
                "boundary")
    mtp_architecture = schema["mtp_architecture"]
    if mtp_architecture == "none":
        if "num_nextn_predict_layers" in cfg:
            raise ValueError(
                f"model_type {model_type!r} does not support MTP fields")
        num_mtp_layers = 0
    elif mtp_architecture == "none_zero_allowed":
        num_mtp_layers = cfg["num_nextn_predict_layers"]
        if num_mtp_layers != 0:
            raise ValueError(
                f"model_type {model_type!r} requires zero MTP layers")
        mtp_architecture = "none"
    elif mtp_architecture in (
            "glm4_moe_mtp", "glm_moe_dsa_mtp", "deepseek_mtp"):
        num_mtp_layers = cfg["num_nextn_predict_layers"]
        if num_mtp_layers == 0:
            mtp_architecture = "none"
    else:
        raise AssertionError(f"invalid MTP schema for {model_type!r}")
    qk_norm_policy = schema["qk_norm"]
    if qk_norm_policy == "config":
        qk_norm = cfg["use_qk_norm"]
        if not isinstance(qk_norm, bool):
            raise TypeError("model config use_qk_norm must be boolean")
    elif isinstance(qk_norm_policy, bool):
        qk_norm = qk_norm_policy
        if ("use_qk_norm" in cfg
                and cfg["use_qk_norm"] is not qk_norm_policy):
            raise ValueError(
                f"model config use_qk_norm disagrees with {model_type!r} "
                "architecture semantics")
    else:
        raise AssertionError(f"invalid Q/K norm schema for {model_type!r}")

    if model_type in ("glm4_moe", "glm_moe_dsa", "kimi_k2", "deepseek_v3"):
        if "routed_scaling_factor" in cfg:
            routed_scaling_factor = cfg["routed_scaling_factor"]
        elif model_type == "glm4_moe":
            # transformers Glm4MoeConfig source default. Normalize it here
            # because this loader deliberately does not instantiate HF config.
            routed_scaling_factor = 1.0
        else:
            raise ValueError(
                f"model_type {model_type!r} requires routed_scaling_factor")
        if (isinstance(routed_scaling_factor, bool)
                or not isinstance(routed_scaling_factor, (int, float))
                or not math.isfinite(routed_scaling_factor)
                or routed_scaling_factor <= 0):
            raise ValueError(
                "model config routed_scaling_factor must be finite and "
                "positive")
    elif model_type in ("qwen3", "qwen3_moe", "llama"):
        if "routed_scaling_factor" in cfg:
            raise ValueError(
                f"model_type {model_type!r} does not support "
                "routed_scaling_factor")
        routed_scaling_factor = 1.0
    else:
        raise AssertionError(
            f"unexpected routed-scaling model_type: {model_type!r}")

    if model_type == "glm_moe_dsa":
        if cfg["moe_router_dtype"] != "float32":
            raise ValueError("GLM-5.2 requires float32 MoE routing")
        rope_parameters = cfg["rope_parameters"]
        if not isinstance(rope_parameters, dict):
            raise TypeError("GLM-5.2 rope_parameters must be a mapping")
        expected_rope_fields = {"rope_theta", "rope_type"}
        if set(rope_parameters) != expected_rope_fields:
            raise ValueError(
                "GLM-5.2 rope_parameters fields must be exactly "
                f"{sorted(expected_rope_fields)}, got "
                f"{sorted(rope_parameters)}")
        raw_index_pattern = cfg["index_topk_pattern"]
        if raw_index_pattern is None:
            index_topk_pattern = None
        elif isinstance(raw_index_pattern, list):
            if len(raw_index_pattern) != cfg["num_hidden_layers"]:
                raise ValueError(
                    "GLM-5.2 index_topk_pattern must cover every layer")
            index_topk_pattern = tuple(raw_index_pattern)
        else:
            raise TypeError(
                "GLM-5.2 index_topk_pattern must be a list or null")
        index_skip_topk_offset = cfg["index_skip_topk_offset"]
        index_topk_frequency = cfg["index_topk_freq"]
        for field, value in (
                ("index_skip_topk_offset", index_skip_topk_offset),
                ("index_topk_freq", index_topk_frequency)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"GLM-5.2 {field} must be an integer")
            if value <= 0:
                raise ValueError(f"GLM-5.2 {field} must be positive")
        if index_topk_pattern is None:
            expected_indexer_type_values = []
            for layer_index in range(cfg["num_hidden_layers"]):
                skipped_count = max(
                    layer_index - index_skip_topk_offset + 1, 0)
                if skipped_count % index_topk_frequency:
                    expected_indexer_type_values.append("shared")
                else:
                    expected_indexer_type_values.append("full")
            expected_indexer_types = tuple(expected_indexer_type_values)
        else:
            expected_indexer_type_values = []
            for value in index_topk_pattern:
                if value == "S":
                    expected_indexer_type_values.append("shared")
                elif value == "F":
                    expected_indexer_type_values.append("full")
                else:
                    raise ValueError(
                        "GLM-5.2 index_topk_pattern values must be S or F, "
                        f"got {value!r}")
            expected_indexer_types = tuple(expected_indexer_type_values)
        if indexer_types != expected_indexer_types:
            raise ValueError(
                "GLM-5.2 indexer_types disagree with vLLM's pinned "
                "index skip rule")
        if cfg["rope_interleave"] is not True:
            raise ValueError("GLM-5.2 requires interleaved main RoPE")
        if cfg["indexer_rope_interleave"] is not True:
            raise ValueError("GLM-5.2 requires interleaved indexer RoPE")
        target_semantics = TargetModelSemantics(
            workload_contract="glm_5_2_vllm_0_27_1_logical_v1",
            use_cache=cfg["use_cache"],
            moe_layer_frequency=cfg["moe_layer_freq"],
            moe_router_dtype="fp32",
            moe_score_function=cfg["scoring_func"],
            moe_topk_method=cfg["topk_method"],
            moe_group_count=cfg["n_group"],
            moe_topk_group_count=cfg["topk_group"],
            moe_normalize_topk_probability=cfg["norm_topk_prob"],
            rope_type=rope_parameters["rope_type"],
            rope_theta=rope_parameters["rope_theta"],
            rope_layout="interleaved",
            rope_factor=None,
            rope_original_max_position_embeddings=None,
            rope_beta_fast=None,
            rope_beta_slow=None,
            rope_mscale=None,
            rope_mscale_all_dim=None,
            index_skip_topk_offset=index_skip_topk_offset,
            index_topk_frequency=index_topk_frequency,
            index_topk_pattern=index_topk_pattern,
            indexer_rope_layout="interleaved",
            mtp_runtime_status=(
                "main_model_skips_spec_layer_weights_no_registered_"
                "glm_moe_dsa_mtp_model"),
        )
    elif model_type == "kimi_k2":
        rope_scaling = cfg["rope_scaling"]
        if not isinstance(rope_scaling, dict):
            raise TypeError("Kimi-K2.6 rope_scaling must be a mapping")
        expected_rope_fields = {
            "beta_fast", "beta_slow", "factor", "mscale",
            "mscale_all_dim", "original_max_position_embeddings", "type",
        }
        if set(rope_scaling) != expected_rope_fields:
            raise ValueError(
                "Kimi-K2.6 rope_scaling fields must be exactly "
                f"{sorted(expected_rope_fields)}, got "
                f"{sorted(rope_scaling)}")
        if rope_scaling["type"] != "yarn":
            raise ValueError("Kimi-K2.6 requires YaRN rope scaling")
        target_semantics = TargetModelSemantics(
            workload_contract="kimi_k2_6_vllm_0_27_1_logical_v1",
            use_cache=cfg["use_cache"],
            moe_layer_frequency=cfg["moe_layer_freq"],
            moe_router_dtype=compute_dtype,
            moe_score_function=cfg["scoring_func"],
            moe_topk_method=cfg["topk_method"],
            moe_group_count=cfg["n_group"],
            moe_topk_group_count=cfg["topk_group"],
            moe_normalize_topk_probability=cfg["norm_topk_prob"],
            rope_type="deepseek_yarn",
            rope_theta=cfg["rope_theta"],
            rope_layout="interleaved",
            rope_factor=rope_scaling["factor"],
            rope_original_max_position_embeddings=(
                rope_scaling["original_max_position_embeddings"]),
            rope_beta_fast=rope_scaling["beta_fast"],
            rope_beta_slow=rope_scaling["beta_slow"],
            rope_mscale=rope_scaling["mscale"],
            rope_mscale_all_dim=rope_scaling["mscale_all_dim"],
            index_skip_topk_offset=None,
            index_topk_frequency=None,
            index_topk_pattern=None,
            indexer_rope_layout=None,
            mtp_runtime_status="not_configured",
        )
    elif model_type == "deepseek_v3":
        rope_scaling = cfg["rope_scaling"]
        if not isinstance(rope_scaling, dict):
            raise TypeError("DeepSeek-V3 rope_scaling must be a mapping")
        expected_rope_fields = {
            "beta_fast", "beta_slow", "factor", "mscale",
            "mscale_all_dim", "original_max_position_embeddings", "type",
        }
        if set(rope_scaling) != expected_rope_fields:
            raise ValueError(
                "DeepSeek-V3 rope_scaling fields must be exactly "
                f"{sorted(expected_rope_fields)}, got "
                f"{sorted(rope_scaling)}")
        if rope_scaling["type"] != "yarn":
            raise ValueError("DeepSeek-V3 requires YaRN rope scaling")
        target_semantics = TargetModelSemantics(
            workload_contract="deepseek_v3_vllm_0_27_1_logical_v1",
            use_cache=cfg["use_cache"],
            moe_layer_frequency=cfg["moe_layer_freq"],
            moe_router_dtype=compute_dtype,
            moe_score_function=cfg["scoring_func"],
            moe_topk_method=cfg["topk_method"],
            moe_group_count=cfg["n_group"],
            moe_topk_group_count=cfg["topk_group"],
            moe_normalize_topk_probability=cfg["norm_topk_prob"],
            rope_type="deepseek_yarn",
            rope_theta=cfg["rope_theta"],
            rope_layout="interleaved",
            rope_factor=rope_scaling["factor"],
            rope_original_max_position_embeddings=(
                rope_scaling["original_max_position_embeddings"]),
            rope_beta_fast=rope_scaling["beta_fast"],
            rope_beta_slow=rope_scaling["beta_slow"],
            rope_mscale=rope_scaling["mscale"],
            rope_mscale_all_dim=rope_scaling["mscale_all_dim"],
            index_skip_topk_offset=None,
            index_topk_frequency=None,
            index_topk_pattern=None,
            indexer_rope_layout=None,
            mtp_runtime_status=(
                "vllm_registered_deepseek_mtp_scheduler_only_draft_contract"),
        )
    elif model_type in ("qwen3", "qwen3_moe", "llama", "glm4_moe"):
        target_semantics = None
    else:
        raise AssertionError(
            f"unhandled target-semantics model_type: {model_type!r}")

    return ModelConfig(
        hidden_size=hidden_size,
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=cfg["num_hidden_layers"],
        head_dim=head_dim,
        vocab_size=cfg["vocab_size"],
        compute_dtype=compute_dtype,
        weight_dtype=weight_dtype,
        weight_group_size=weight_group_size,
        quantized_weight_targets=quantized_weight_targets,
        quantized_weight_ignore=quantized_weight_ignore,
        attention_layout=attention_layout,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        indexer_types=indexer_types,
        index_share_for_mtp_iteration=index_share_for_mtp_iteration,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        architecture=architecture,
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=attention_bias_field,
        mlp_bias=mlp_bias_field,
        max_position_embeddings=cfg["max_position_embeddings"],
        model_type=model_type,
        num_dense_layers=num_dense_layers,
        num_shared_experts=num_shared_experts,
        num_mtp_layers=num_mtp_layers,
        mtp_architecture=mtp_architecture,
        qk_norm=qk_norm,
        rope_scaling_type=_rope_scaling_type(cfg),
        routed_scaling_factor=float(routed_scaling_factor),
        target_semantics=target_semantics,
        config_sha256=config_sha256,
        source_config_sha256=source_config_sha256,
        canonical_config_sha256=canonical_config_sha256,
        config_canonicalization=_CONFIG_CANONICALIZATION,
        config_extraction_path=config_extraction_path,
        extracted_config_sha256=extracted_config_sha256,
        model_id=model_id,
        model_revision=model_revision,
        model_source_url=model_source_url,
        model_scope=model_scope,
    )
