"""Frozen model constants for the three OrbitServe serving targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelServingProfile:
    name: str
    family: str
    total_parameter_count: int
    active_parameter_count: int
    layer_count: int
    hidden_size: int
    expert_count: int
    prefill_linear_flops_per_token: float
    prefill_quadratic_flops_coefficient: float
    kv_cache_bytes_per_token: float
    max_context_tokens: int
    weight_bytes_per_parameter: float = 2.0
    activation_bytes: float = 2.0
    architecture_source: str = ""
    approximation_notes: tuple[str, ...] = ()

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 1

    @property
    def total_weight_bytes(self) -> float:
        return self.total_parameter_count * self.weight_bytes_per_parameter

    @property
    def active_weight_bytes(self) -> float:
        return self.active_parameter_count * self.weight_bytes_per_parameter


MODEL_SERVING_PROFILES: dict[str, ModelServingProfile] = {
    "llama3.3-70b": ModelServingProfile(
        name="llama3.3-70b",
        family="llama3",
        total_parameter_count=70_000_000_000,
        active_parameter_count=70_000_000_000,
        layer_count=80,
        hidden_size=8192,
        expert_count=1,
        prefill_linear_flops_per_token=136_902_082_560.0,
        prefill_quadratic_flops_coefficient=2_621_440.0,
        kv_cache_bytes_per_token=327_680.0,
        max_context_tokens=131_072,
        architecture_source="meta-llama/Llama-3.3-70B-Instruct config",
    ),
    "deepseek-v3": ModelServingProfile(
        name="deepseek-v3",
        family="deepseek",
        total_parameter_count=671_000_000_000,
        active_parameter_count=37_000_000_000,
        layer_count=61,
        hidden_size=7168,
        expert_count=256,
        prefill_linear_flops_per_token=71_182_974_976.0,
        prefill_quadratic_flops_coefficient=4_997_120.0,
        kv_cache_bytes_per_token=70_272.0,
        max_context_tokens=128_000,
        architecture_source="deepseek-ai/DeepSeek-V3 configuration_deepseek.py",
        approximation_notes=("MLA KV stores the compressed latent and RoPE key",),
    ),
    "kimi-k3": ModelServingProfile(
        name="kimi-k3",
        family="kimi",
        total_parameter_count=2_800_000_000_000,
        active_parameter_count=104_000_000_000,
        layer_count=93,
        hidden_size=7168,
        expert_count=896,
        prefill_linear_flops_per_token=208_000_000_000.0,
        prefill_quadratic_flops_coefficient=1_474_560.0,
        kv_cache_bytes_per_token=27_648.0,
        max_context_tokens=1_048_576,
        architecture_source="moonshotai/Kimi-K3 config and technical report",
        approximation_notes=(
            "quadratic term covers the 24 full MLA layers",
            "KDA fixed recurrent state is excluded from per-token KV bytes",
            "text-only serving excludes the vision tower",
        ),
    ),
}


MODEL_NAME_ALIASES = {
    "llama-3.3-70b": "llama3.3-70b",
    "llama3.3-70b-instruct": "llama3.3-70b",
    "deepseek-v3-0324": "deepseek-v3",
    "kimi-k3-instruct": "kimi-k3",
}


def model_serving_profile(name: str) -> ModelServingProfile:
    normalized = str(name).strip().lower()
    normalized = MODEL_NAME_ALIASES.get(normalized, normalized)
    try:
        return MODEL_SERVING_PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_SERVING_PROFILES))
        raise ValueError(f"unsupported serving model {name!r}; choose one of: {supported}") from exc


__all__ = ["MODEL_SERVING_PROFILES", "ModelServingProfile", "model_serving_profile"]
