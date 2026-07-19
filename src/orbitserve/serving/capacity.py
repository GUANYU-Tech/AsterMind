"""LLM serving throughput models for OrbitServe experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from ..common import to_jsonable
from ..architecture import SatelliteHardwareProfile
from ..common import PerformanceMetrics
from ..workload import TokenDistribution


BYTES_PER_GB = 1e9


def _pick(data: dict[str, Any], key: str, default: Any) -> Any:
    return data[key] if key in data else default






@dataclass(frozen=True)
class PrecisionSpec:
    name: str = "bf16"
    weight_bits: int = 16
    activation_bits: int = 16
    kv_cache_bits: int = 16

    @classmethod
    def from_value(cls, value: Any) -> "PrecisionSpec":
        if isinstance(value, PrecisionSpec):
            return value
        if isinstance(value, dict):
            return cls(
                name=str(_pick(value, "name", "custom")),
                weight_bits=int(_pick(value, "weight_bits", 16)),
                activation_bits=int(_pick(value, "activation_bits", 16)),
                kv_cache_bits=int(_pick(value, "kv_cache_bits", _pick(value, "activation_bits", 16))),
            )
        name = str(value or "bf16").lower()
        presets = {
            "fp32": cls("fp32", 32, 32, 32),
            "fp16": cls("fp16", 16, 16, 16),
            "bf16": cls("bf16", 16, 16, 16),
            "fp8": cls("fp8", 8, 8, 8),
            "int8": cls("int8", 8, 8, 8),
            "int4": cls("int4", 4, 8, 8),
        }
        if name not in presets:
            raise ValueError(f"unknown precision '{name}'; use bf16/fp16/fp8/int8/int4 or a custom dict")
        return presets[name]

    @property
    def weight_bytes(self) -> float:
        return self.weight_bits / 8.0

    @property
    def activation_bytes(self) -> float:
        return self.activation_bits / 8.0

    @property
    def kv_cache_bytes(self) -> float:
        return self.kv_cache_bits / 8.0

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class LLMModelProfile:
    name: str = "llama-like-34b"
    parameter_count_b: float = 34.0
    precision: PrecisionSpec = field(default_factory=PrecisionSpec)
    max_context_tokens: int = 8192
    family: str = "llama"
    variant: str | None = None
    active_parameter_count_b: float | None = None
    layer_count: int | None = None
    hidden_size: int | None = None
    attention_head_count: int | None = None
    kv_head_count: int | None = None
    intermediate_size: int | None = None
    non_routed_parameter_count_b: float | None = None
    routed_expert_parameter_count_b: float | None = None
    moe_layer_count: int | None = None
    expert_count: int | None = None
    active_expert_count: int | None = None
    architecture_source: str = "configured"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LLMModelProfile":
        data = data or {}
        preset_name = data.get("preset")
        preset_data = _model_preset(str(preset_name)) if preset_name is not None else {}
        merged = {**preset_data, **data}
        return cls(
            name=str(_pick(merged, "name", _pick(preset_data, "name", "llama-like-34b"))),
            parameter_count_b=float(_pick(merged, "parameter_count_b", 34.0)),
            precision=PrecisionSpec.from_value(_pick(merged, "precision", "bf16")),
            max_context_tokens=int(_pick(merged, "max_context_tokens", 8192)),
            family=str(_pick(merged, "family", "llama")),
            variant=_optional_str(merged.get("variant")),
            active_parameter_count_b=_optional_float(merged.get("active_parameter_count_b")),
            layer_count=_optional_int(merged.get("layer_count")),
            hidden_size=_optional_int(merged.get("hidden_size")),
            attention_head_count=_optional_int(merged.get("attention_head_count")),
            kv_head_count=_optional_int(merged.get("kv_head_count")),
            intermediate_size=_optional_int(merged.get("intermediate_size")),
            non_routed_parameter_count_b=_optional_float(merged.get("non_routed_parameter_count_b")),
            routed_expert_parameter_count_b=_optional_float(merged.get("routed_expert_parameter_count_b")),
            moe_layer_count=_optional_int(merged.get("moe_layer_count")),
            expert_count=_optional_int(merged.get("expert_count")),
            active_expert_count=_optional_int(merged.get("active_expert_count")),
            architecture_source=str(_pick(merged, "architecture_source", "configured")),
        ).resolved()

    @property
    def active_parameters_b(self) -> float:
        return self.active_parameter_count_b or self.parameter_count_b

    @property
    def is_moe(self) -> bool:
        return bool(
            self.expert_count
            and self.active_expert_count
            and self.expert_count > 0
            and self.active_expert_count > 0
        )

    @property
    def resolved_moe_layer_count(self) -> int:
        return self.moe_layer_count or self.layer_count or 0

    @property
    def routed_expert_parameters_b(self) -> float:
        if self.routed_expert_parameter_count_b is not None:
            return self.routed_expert_parameter_count_b
        split = self._derived_moe_split()
        return split["routed_expert_parameter_count_b"] if split else 0.0

    @property
    def non_routed_parameters_b(self) -> float:
        if self.non_routed_parameter_count_b is not None:
            return self.non_routed_parameter_count_b
        split = self._derived_moe_split()
        if split:
            return split["non_routed_parameter_count_b"]
        return self.parameter_count_b

    @property
    def active_routed_parameters_b(self) -> float:
        return max(0.0, self.active_parameters_b - self.non_routed_parameters_b)

    @property
    def routed_expert_memory_gb(self) -> float:
        return self.routed_expert_parameters_b * 1e9 * self.precision.weight_bytes / BYTES_PER_GB

    @property
    def non_routed_memory_gb(self) -> float:
        return self.non_routed_parameters_b * 1e9 * self.precision.weight_bytes / BYTES_PER_GB

    @property
    def active_routed_memory_gb(self) -> float:
        return self.active_routed_parameters_b * 1e9 * self.precision.weight_bytes / BYTES_PER_GB

    def _derived_moe_split(self) -> dict[str, float] | None:
        if not self.is_moe or self.active_parameter_count_b is None:
            return None
        expert_count = self.expert_count or 0
        active_experts = self.active_expert_count or 0
        if expert_count <= active_experts:
            return None
        per_expert = (self.parameter_count_b - self.active_parameter_count_b) / (
            expert_count - active_experts
        )
        routed = expert_count * per_expert
        non_routed = self.parameter_count_b - routed
        if per_expert < 0 or routed < 0 or non_routed < 0:
            return None
        return {
            "routed_expert_parameter_count_b": routed,
            "non_routed_parameter_count_b": non_routed,
            "routed_parameter_count_b_per_expert": per_expert,
        }

    @property
    def head_dim(self) -> float:
        if not self.hidden_size or not self.attention_head_count:
            return 0.0
        return self.hidden_size / self.attention_head_count

    @property
    def weight_memory_gb(self) -> float:
        return self.parameter_count_b * 1e9 * self.precision.weight_bytes / BYTES_PER_GB

    @property
    def kv_cache_bytes_per_token(self) -> float:
        if not self.layer_count or not self.kv_head_count:
            return 0.0
        return (
            2.0
            * self.layer_count
            * self.kv_head_count
            * self.head_dim
            * self.precision.kv_cache_bytes
        )

    def prefill_flops(self, prompt_tokens: int) -> float:
        gemm = 2.0 * self.active_parameters_b * 1e9 * prompt_tokens
        attention = 4.0 * (self.layer_count or 0) * (self.hidden_size or 0) * prompt_tokens**2
        return gemm + attention

    def prefill_flop_breakdown(self, prompt_tokens: int) -> dict[str, float]:
        non_routed_gemm = 2.0 * self.non_routed_parameters_b * 1e9 * prompt_tokens
        expert_gemm = 2.0 * self.active_routed_parameters_b * 1e9 * prompt_tokens
        attention = 4.0 * (self.layer_count or 0) * (self.hidden_size or 0) * prompt_tokens**2
        return {
            "non_routed": non_routed_gemm + attention,
            "routed_expert": expert_gemm,
            "attention": attention,
        }

    def decode_flops_per_token(self, context_tokens: int) -> float:
        gemm = 2.0 * self.active_parameters_b * 1e9
        attention = 4.0 * (self.layer_count or 0) * (self.hidden_size or 0) * context_tokens
        return gemm + attention

    def decode_flop_breakdown_per_token(self, context_tokens: int) -> dict[str, float]:
        non_routed_gemm = 2.0 * self.non_routed_parameters_b * 1e9
        expert_gemm = 2.0 * self.active_routed_parameters_b * 1e9
        attention = 4.0 * (self.layer_count or 0) * (self.hidden_size or 0) * context_tokens
        return {
            "non_routed": non_routed_gemm + attention,
            "routed_expert": expert_gemm,
            "attention": attention,
        }

    def resolved(self) -> "LLMModelProfile":
        missing = [
            self.layer_count is None,
            self.hidden_size is None,
            self.attention_head_count is None,
            self.kv_head_count is None,
        ]
        if not any(missing):
            return self
        heuristic = _heuristic_architecture(self.parameter_count_b, self.family)
        return replace(
            self,
            layer_count=self.layer_count or heuristic["layer_count"],
            hidden_size=self.hidden_size or heuristic["hidden_size"],
            attention_head_count=self.attention_head_count or heuristic["attention_head_count"],
            kv_head_count=self.kv_head_count or heuristic["kv_head_count"],
            intermediate_size=self.intermediate_size or heuristic["intermediate_size"],
            moe_layer_count=self.moe_layer_count,
            architecture_source=(
                self.architecture_source
                if self.architecture_source != "configured"
                else "heuristic_from_parameter_count"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "variant": self.variant,
            "parameter_count_b": self.parameter_count_b,
            "active_parameter_count_b": self.active_parameter_count_b,
            "precision": self.precision.to_dict(),
            "max_context_tokens": self.max_context_tokens,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "attention_head_count": self.attention_head_count,
            "kv_head_count": self.kv_head_count,
            "intermediate_size": self.intermediate_size,
            "non_routed_parameter_count_b": self.non_routed_parameter_count_b,
            "routed_expert_parameter_count_b": self.routed_expert_parameter_count_b,
            "derived_non_routed_parameter_count_b": self.non_routed_parameters_b,
            "derived_routed_expert_parameter_count_b": self.routed_expert_parameters_b,
            "active_routed_parameter_count_b": self.active_routed_parameters_b,
            "moe_layer_count": self.moe_layer_count,
            "resolved_moe_layer_count": self.resolved_moe_layer_count,
            "expert_count": self.expert_count,
            "active_expert_count": self.active_expert_count,
            "architecture_source": self.architecture_source,
            "weight_memory_gb": self.weight_memory_gb,
            "non_routed_memory_gb": self.non_routed_memory_gb,
            "routed_expert_memory_gb": self.routed_expert_memory_gb,
            "active_routed_memory_gb": self.active_routed_memory_gb,
            "kv_cache_bytes_per_token": self.kv_cache_bytes_per_token,
        }


@dataclass(frozen=True)
class StageServingProfile:
    token_throughput_per_s_per_accelerator: float | None = None
    power_fraction: float | None = None
    batching_window_ms: float = 20.0
    batch_size: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StageServingProfile":
        data = data or {}
        return cls(
            token_throughput_per_s_per_accelerator=_optional_float(
                data.get("token_throughput_per_s_per_accelerator")
            ),
            power_fraction=_optional_float(data.get("power_fraction")),
            batching_window_ms=float(_pick(data, "batching_window_ms", 20.0)),
            batch_size=max(1, int(_pick(data, "batch_size", 1))),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class ServingParallelism:
    strategy: str = "single_satellite"
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    expert_parallel_degree: int = 1
    data_parallel_degree: int = 1
    stage_dp_mode: str = "none"
    dp_attention: int | None = None
    dp_expert: int | None = None
    microbatch_count: int = 1
    max_satellites: int = 1
    auto_select: bool = False
    tensor_parallel_collectives_per_layer: int = 2
    expert_parallel_imbalance_factor: float = 1.0
    expert_parallel_startup_us: float = 0.0
    moe_imbalance_factor: float = 1.0
    alpha_ep: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ServingParallelism":
        data = data or {}
        strategy = str(_pick(data, "strategy", "single_satellite"))
        stage_dp_mode = str(_pick(data, "stage_dp_mode", "none")).lower()
        if stage_dp_mode not in ("none", "uniform", "path_specific"):
            raise ValueError("stage_dp_mode must be one of none, uniform, path_specific")
        ep_imbalance = max(
            1.0,
            float(_pick(data, "expert_parallel_imbalance_factor", _pick(data, "moe_imbalance_factor", 1.0))),
        )
        return cls(
            strategy=strategy,
            tensor_parallel_degree=max(1, int(_pick(data, "tensor_parallel_degree", _pick(data, "tp", 1)))),
            pipeline_parallel_degree=max(1, int(_pick(data, "pipeline_parallel_degree", _pick(data, "pp", 1)))),
            expert_parallel_degree=max(1, int(_pick(data, "expert_parallel_degree", _pick(data, "ep", 1)))),
            data_parallel_degree=max(1, int(_pick(data, "data_parallel_degree", _pick(data, "dp", 1)))),
            stage_dp_mode=stage_dp_mode,
            dp_attention=_optional_int(data.get("dp_attention")),
            dp_expert=_optional_int(data.get("dp_expert")),
            microbatch_count=max(1, int(_pick(data, "microbatch_count", 1))),
            max_satellites=max(1, int(_pick(data, "max_satellites", 1))),
            auto_select=bool(_pick(data, "auto_select", strategy == "auto")),
            tensor_parallel_collectives_per_layer=max(
                0, int(_pick(data, "tensor_parallel_collectives_per_layer", 2))
            ),
            expert_parallel_imbalance_factor=ep_imbalance,
            expert_parallel_startup_us=max(0.0, float(_pick(data, "expert_parallel_startup_us", 0.0))),
            moe_imbalance_factor=ep_imbalance,
            alpha_ep=max(0.0, float(_pick(data, "alpha_ep", 0.0))),
        )

    @property
    def dp(self) -> int:
        return self.data_parallel_degree

    @property
    def effective_dp_attention(self) -> int:
        if self.stage_dp_mode == "none":
            return 1
        if self.stage_dp_mode == "uniform":
            return self.data_parallel_degree
        return max(1, self.dp_attention or self.data_parallel_degree)

    @property
    def effective_dp_expert(self) -> int:
        if self.stage_dp_mode == "none":
            return 1
        if self.stage_dp_mode == "uniform":
            return self.data_parallel_degree
        return max(1, self.dp_expert or self.data_parallel_degree)

    @property
    def stage_parallel_degree(self) -> int:
        if self.stage_dp_mode == "uniform":
            return self.data_parallel_degree * math.lcm(
                self.tensor_parallel_degree,
                self.expert_parallel_degree,
            )
        if self.stage_dp_mode == "path_specific":
            return math.lcm(
                self.tensor_parallel_degree * self.effective_dp_attention,
                self.expert_parallel_degree * self.effective_dp_expert,
            )
        return math.lcm(self.tensor_parallel_degree, self.expert_parallel_degree)

    @property
    def satellite_count(self) -> int:
        return self.stage_parallel_degree * self.pipeline_parallel_degree

    @property
    def bubble_multiplier(self) -> float:
        if self.pipeline_parallel_degree <= 1:
            return 1.0
        return (self.microbatch_count + self.pipeline_parallel_degree - 1) / self.microbatch_count

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["tp"] = self.tensor_parallel_degree
        payload["pp"] = self.pipeline_parallel_degree
        payload["ep"] = self.expert_parallel_degree
        payload["dp"] = self.data_parallel_degree
        payload["dp_attention_effective"] = self.effective_dp_attention
        payload["dp_expert_effective"] = self.effective_dp_expert
        payload["stage_parallel_degree"] = self.stage_parallel_degree
        payload["satellite_count"] = self.satellite_count
        payload["bubble_multiplier"] = self.bubble_multiplier
        return payload


@dataclass(frozen=True)
class ParallelLayout:
    stage_satellite_count: int
    satellites_per_deployment: int
    dp: int
    dp_attention: int
    dp_expert: int
    dp_replica_factor: int
    dp_replica_model: str
    latency_parallelism_always_path: int
    latency_parallelism_routed_path: int
    throughput_parallelism_always_path: int
    throughput_parallelism_routed_path: int
    placement_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class ServingSpec:
    model: LLMModelProfile = field(default_factory=LLMModelProfile)
    prefill: StageServingProfile = field(default_factory=StageServingProfile)
    decode: StageServingProfile = field(default_factory=StageServingProfile)
    parallelism: ServingParallelism = field(default_factory=ServingParallelism)
    pd_mode: str = "colocated"
    prefill_satellite_fraction: float | None = None
    decode_satellite_fraction: float | None = None
    hardware: SatelliteHardwareProfile | None = None
    kv_cache_capacity_gb: float | None = None
    effective_utilization: float = 0.5
    effective_memory_utilization: float = 0.7
    interconnect_utilization: float = 0.7
    memory_reserve_fraction: float = 0.1
    pd_placement: dict[str, Any] = field(default_factory=dict)
    satellite_budget: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ServingSpec":
        data = data or {}
        pd_mode = str(_pick(data, "pd_mode", "colocated")).lower()
        if pd_mode not in ("colocated", "disaggregated"):
            raise ValueError("pd_mode must be one of colocated, disaggregated")
        prefill_satellite_fraction = _optional_float(data.get("prefill_satellite_fraction"))
        decode_satellite_fraction = _optional_float(data.get("decode_satellite_fraction"))
        _resolved_pd_satellite_fractions(
            pd_mode=pd_mode,
            prefill_fraction=prefill_satellite_fraction,
            decode_fraction=decode_satellite_fraction,
            fail_on_invalid=True,
        )
        return cls(
            model=LLMModelProfile.from_dict(data.get("model")),
            prefill=StageServingProfile.from_dict(data.get("prefill")),
            decode=StageServingProfile.from_dict(data.get("decode")),
            parallelism=ServingParallelism.from_dict(data.get("parallelism")),
            pd_mode=pd_mode,
            prefill_satellite_fraction=prefill_satellite_fraction,
            decode_satellite_fraction=decode_satellite_fraction,
            hardware=SatelliteHardwareProfile.from_dict(data["hardware"]) if data.get("hardware") else None,
            kv_cache_capacity_gb=_optional_float(data.get("kv_cache_capacity_gb")),
            effective_utilization=float(_pick(data, "effective_utilization", 0.5)),
            effective_memory_utilization=float(_pick(data, "effective_memory_utilization", 0.7)),
            interconnect_utilization=float(_pick(data, "interconnect_utilization", 0.7)),
            memory_reserve_fraction=float(_pick(data, "memory_reserve_fraction", 0.1)),
            pd_placement=dict(data.get("pd_placement") or {}),
            satellite_budget=dict(data.get("satellite_budget") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "prefill": self.prefill.to_dict(),
            "decode": self.decode.to_dict(),
            "parallelism": self.parallelism.to_dict(),
            "pd_mode": self.pd_mode,
            "prefill_satellite_fraction": self.prefill_satellite_fraction,
            "decode_satellite_fraction": self.decode_satellite_fraction,
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "kv_cache_capacity_gb": self.kv_cache_capacity_gb,
            "effective_utilization": self.effective_utilization,
            "effective_memory_utilization": self.effective_memory_utilization,
            "interconnect_utilization": self.interconnect_utilization,
            "memory_reserve_fraction": self.memory_reserve_fraction,
            "pd_placement": to_jsonable(self.pd_placement),
            "satellite_budget": to_jsonable(self.satellite_budget),
        }


@dataclass(frozen=True)
class ServingCapacityProfile:
    evaluated: bool
    feasible: bool
    deployment_mode: str
    pd_mode: str
    model: dict[str, Any]
    hardware: dict[str, Any]
    parallelism: dict[str, Any]
    stage_satellite_count: int
    satellites_per_deployment: int
    dp: int
    dp_attention: int
    dp_expert: int
    dp_replica_factor: int
    latency_parallelism_always_path: int
    latency_parallelism_routed_path: int
    throughput_parallelism_always_path: int
    throughput_parallelism_routed_path: int
    satellite_count_used: int
    serving_satellite_budget_count: int | None
    constellation_satellite_count: int | None
    satellite_budget_mode: str | None
    comparison_group_id: str | None
    deployment_group_count: int
    prefill_deployment_group_count: int
    decode_deployment_group_count: int
    local_tensor_parallel_degree: int
    data_parallel_replicas_per_group: int
    total_data_parallel_replicas: int
    min_satellites_for_weights: int
    weight_memory_gb: float
    weight_memory_per_satellite_gb: float
    usable_memory_per_satellite_gb: float
    kv_cache_bytes_per_token: float
    kv_cache_capacity_tokens: int | None
    prefill_tokens_per_s: float | None
    decode_tokens_per_s: float | None
    request_throughput_per_s: float | None
    ttft_s: float | None
    e2e_s: float | None
    prefill_interconnect_time_s: float | None
    decode_interconnect_time_s: float | None
    effective_isl_bandwidth_gbps: float | None
    isl_required_flows: int
    available_isl_laser_heads: int
    isl_head_occupancy: float | None
    tp_communication_time_s: float | None
    ep_communication_time_s: float | None
    pp_communication_time_s: float | None
    kv_handoff_time_s: float | None
    communication_bottleneck: str | None
    placement_warnings: tuple[str, ...]
    bottlenecks: dict[str, str]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_performance_metrics(self) -> PerformanceMetrics:
        if not self.evaluated:
            return PerformanceMetrics()
        return PerformanceMetrics(
            evaluated=True,
            request_throughput_per_s=self.request_throughput_per_s,
            token_throughput_per_s=self.decode_tokens_per_s,
            ttft_p95_s=None,
            e2e_p95_s=None,
            prefill_tokens_per_s=self.prefill_tokens_per_s,
            decode_tokens_per_s=self.decode_tokens_per_s,
            model_weight_memory_gb=self.weight_memory_gb,
            kv_cache_capacity_tokens=self.kv_cache_capacity_tokens,
            satellite_count_used=self.satellite_count_used,
            feasible=self.feasible,
            bottleneck=", ".join(f"{key}:{value}" for key, value in sorted(self.bottlenecks.items())),
            details=self.to_dict(),
            warnings=self.warnings,
        )


def estimate_serving_throughput(
    serving: ServingSpec,
    *,
    hardware: SatelliteHardwareProfile | None = None,
    token_distribution: TokenDistribution | None = None,
    constellation_satellite_count: int | None = None,
) -> ServingCapacityProfile:
    """Estimate LLM prefill/decode throughput for a hardware profile."""

    resolved_hardware = hardware or serving.hardware
    if resolved_hardware is None:
        return _unevaluated_throughput("serving.hardware is not configured")
    model = serving.model.resolved()
    token_distribution = token_distribution or TokenDistribution()
    parallelism, parallelism_warnings = _resolve_parallelism(
        serving.parallelism,
        model,
        resolved_hardware,
        constellation_satellite_count=constellation_satellite_count,
        memory_reserve_fraction=serving.memory_reserve_fraction,
    )

    warnings = list(parallelism_warnings)
    if model.architecture_source == "heuristic_from_parameter_count":
        warnings.append(
            f"{model.name} architecture was inferred from parameter_count_b; configure layer/head fields for paper runs"
        )
    if model.active_parameter_count_b is not None and model.active_parameter_count_b > model.parameter_count_b:
        warnings.append("active_parameter_count_b exceeds parameter_count_b")
    if model.is_moe and model.active_parameter_count_b is None:
        warnings.append("MoE model is missing active_parameter_count_b; active FLOPs fall back to total parameters")
    if token_distribution.prompt_tokens_p95 > model.max_context_tokens:
        warnings.append("workload prompt_tokens_p95 exceeds model max_context_tokens")

    layout = _parallel_layout(model, parallelism)
    warnings.extend(layout.placement_warnings)
    warnings.extend(_parallelism_model_warnings(model, parallelism))
    pool_plan = _pd_pool_plan(
        serving,
        satellites_per_deployment=layout.satellites_per_deployment,
        constellation_satellite_count=constellation_satellite_count,
    )
    warnings.extend(pool_plan["warnings"])
    satellite_count_used = pool_plan["satellite_count_used"]
    usable_memory_per_sat = resolved_hardware.total_fast_mem_gb * (1.0 - serving.memory_reserve_fraction)
    weight_memory_gb = model.weight_memory_gb
    weight_breakdown = _weight_memory_breakdown_gb(model, parallelism)
    weight_memory_per_sat = weight_breakdown["total"]
    local_tensor_degree = 1
    replicas_per_group = 1
    deployment_group_count = pool_plan["deployment_group_count"]
    prefill_deployment_group_count = pool_plan["prefill_deployment_group_count"]
    decode_deployment_group_count = pool_plan["decode_deployment_group_count"]
    total_data_parallel_replicas = layout.dp_replica_factor * deployment_group_count
    kv_cache_bytes_per_token = model.kv_cache_bytes_per_token
    kv_capacity_tokens = _kv_cache_capacity_tokens(
        serving,
        model,
        parallelism,
        usable_memory_per_sat,
        weight_memory_per_sat,
    )
    min_sats_for_weights = _min_satellites_for_weights(model, usable_memory_per_sat)
    feasible = True
    if constellation_satellite_count is not None and satellite_count_used > constellation_satellite_count:
        feasible = False
        warnings.append(
            f"parallelism uses {satellite_count_used} satellites but constellation has {constellation_satellite_count}"
        )
    if serving.pd_mode == "disaggregated" and (
        prefill_deployment_group_count <= 0 or decode_deployment_group_count <= 0
    ):
        feasible = False
        warnings.append(
            "pd_mode=disaggregated requires at least one prefill and one decode deployment group"
        )
    if parallelism.expert_parallel_degree > 1 and not model.is_moe:
        warnings.append("expert_parallel_degree is configured for a dense/non-MoE model and has no expert weights to shard")
    if model.is_moe and model.expert_count is not None and parallelism.expert_parallel_degree > model.expert_count:
        feasible = False
        warnings.append(
            f"expert_parallel_degree {parallelism.expert_parallel_degree} exceeds expert_count {model.expert_count}"
        )
    if model.is_moe and model._derived_moe_split() is None and (
        model.non_routed_parameter_count_b is None or model.routed_expert_parameter_count_b is None
    ):
        feasible = False
        warnings.append(
            "MoE parameter split could not be derived; configure non_routed_parameter_count_b "
            "and routed_expert_parameter_count_b"
        )
    if weight_memory_per_sat > usable_memory_per_sat:
        feasible = False
        warnings.append(
            f"model weights need {weight_memory_per_sat:.3f} GB per satellite, "
            f"above usable memory {usable_memory_per_sat:.3f} GB"
        )
    decode_context_tokens = min(
        model.max_context_tokens,
        token_distribution.prompt_tokens_p95 + max(1, token_distribution.output_tokens_p95 // 2),
    )
    representative_context = token_distribution.prompt_tokens_p95 + token_distribution.output_tokens_p95
    if kv_capacity_tokens is not None and kv_capacity_tokens < representative_context:
        warnings.append(
            f"KV cache capacity {kv_capacity_tokens} tokens is below p95 context+output {representative_context}"
        )

    prefill = _estimate_prefill(
        serving,
        model,
        resolved_hardware,
        parallelism,
        layout,
        prompt_tokens=token_distribution.prompt_tokens_p95,
        replicas_per_group=replicas_per_group,
        deployment_group_count=prefill_deployment_group_count,
        pd_handoff_active=False,
    )
    decode = _estimate_decode(
        serving,
        model,
        resolved_hardware,
        parallelism,
        layout,
        context_tokens=decode_context_tokens,
        replicas_per_group=replicas_per_group,
        deployment_group_count=decode_deployment_group_count,
        pd_handoff_active=False,
    )
    kv_handoff = _kv_handoff_profile(
        serving,
        model,
        resolved_hardware,
        parallelism,
        prompt_tokens_latency=token_distribution.prompt_tokens_p95,
        prompt_token_count_for_throughput=token_distribution.prompt_tokens_p50,
        base_required_flows=max(prefill["isl_required_flows"], decode["isl_required_flows"]),
    )
    feasible = feasible and math.isfinite(prefill["latency_s"]) and math.isfinite(decode["latency_s"])
    warnings.extend(prefill["warnings"])
    warnings.extend(decode["warnings"])
    warnings.extend(kv_handoff["warnings"])
    if serving.pd_mode == "disaggregated" and not math.isfinite(kv_handoff["latency_s"]):
        feasible = False

    prefill_tokens_per_s = prefill["token_throughput_per_s"]
    decode_tokens_per_s = decode["token_throughput_per_s"]
    request_throughput = None
    if prefill_tokens_per_s and decode_tokens_per_s:
        prefill_request_throughput_per_s = prefill_tokens_per_s / max(1, token_distribution.prompt_tokens_p50)
        decode_request_throughput_per_s = decode_tokens_per_s / max(1, token_distribution.output_tokens_p50)
        if serving.pd_mode == "disaggregated":
            candidates = [prefill_request_throughput_per_s, decode_request_throughput_per_s]
            if kv_handoff["kv_handoff_request_limit_per_s"] is not None:
                candidates.append(kv_handoff["kv_handoff_request_limit_per_s"])
            request_throughput = min(candidates)
        else:
            request_throughput = 1.0 / (
                max(1, token_distribution.prompt_tokens_p50) / prefill_tokens_per_s
                + max(1, token_distribution.output_tokens_p50) / decode_tokens_per_s
            )
    ttft_s = (
        None
        if not math.isfinite(prefill["latency_s"]) or not math.isfinite(kv_handoff["latency_s"])
        else serving.prefill.batching_window_ms / 1000.0 + prefill["latency_s"] + kv_handoff["latency_s"]
    )
    e2e_s = None if ttft_s is None or not math.isfinite(decode["latency_s"]) else (
        ttft_s + token_distribution.output_tokens_p95 * decode["latency_s"]
    )
    deployment_mode = _deployment_mode(parallelism)
    communication_summary = _merge_communication_summaries(prefill, decode, kv_handoff)
    return ServingCapacityProfile(
        evaluated=True,
        feasible=feasible,
        deployment_mode=deployment_mode,
        pd_mode=serving.pd_mode,
        model=model.to_dict(),
        hardware=resolved_hardware.to_dict(),
        parallelism=parallelism.to_dict(),
        stage_satellite_count=layout.stage_satellite_count,
        satellites_per_deployment=layout.satellites_per_deployment,
        dp=layout.dp,
        dp_attention=layout.dp_attention,
        dp_expert=layout.dp_expert,
        dp_replica_factor=layout.dp_replica_factor,
        latency_parallelism_always_path=layout.latency_parallelism_always_path,
        latency_parallelism_routed_path=layout.latency_parallelism_routed_path,
        throughput_parallelism_always_path=layout.throughput_parallelism_always_path,
        throughput_parallelism_routed_path=layout.throughput_parallelism_routed_path,
        satellite_count_used=satellite_count_used,
        serving_satellite_budget_count=pool_plan.get("serving_satellite_budget_count"),
        constellation_satellite_count=constellation_satellite_count,
        satellite_budget_mode=pool_plan.get("satellite_budget_mode"),
        comparison_group_id=pool_plan.get("comparison_group_id"),
        deployment_group_count=deployment_group_count,
        prefill_deployment_group_count=prefill_deployment_group_count,
        decode_deployment_group_count=decode_deployment_group_count,
        local_tensor_parallel_degree=local_tensor_degree,
        data_parallel_replicas_per_group=layout.dp_replica_factor,
        total_data_parallel_replicas=total_data_parallel_replicas,
        min_satellites_for_weights=min_sats_for_weights,
        weight_memory_gb=weight_memory_gb,
        weight_memory_per_satellite_gb=weight_memory_per_sat,
        usable_memory_per_satellite_gb=usable_memory_per_sat,
        kv_cache_bytes_per_token=kv_cache_bytes_per_token,
        kv_cache_capacity_tokens=kv_capacity_tokens,
        prefill_tokens_per_s=prefill_tokens_per_s,
        decode_tokens_per_s=decode_tokens_per_s,
        request_throughput_per_s=request_throughput,
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        prefill_interconnect_time_s=prefill["interconnect_time_s"],
        decode_interconnect_time_s=decode["interconnect_time_s"],
        effective_isl_bandwidth_gbps=communication_summary["effective_isl_bandwidth_gbps"],
        isl_required_flows=communication_summary["isl_required_flows"],
        available_isl_laser_heads=communication_summary["available_isl_laser_heads"],
        isl_head_occupancy=communication_summary["isl_head_occupancy"],
        tp_communication_time_s=communication_summary["tp_communication_time_s"],
        ep_communication_time_s=communication_summary["ep_communication_time_s"],
        pp_communication_time_s=communication_summary["pp_communication_time_s"],
        kv_handoff_time_s=kv_handoff["latency_s"],
        communication_bottleneck=communication_summary["communication_bottleneck"],
        placement_warnings=layout.placement_warnings,
        bottlenecks={
            "prefill": prefill["bottleneck"],
            "decode": decode["bottleneck"],
            "kv_handoff": kv_handoff["bottleneck"],
        },
        warnings=tuple(dict.fromkeys(warnings)),
        provenance={
            "modeling_doc": "docs/research/parameters/llm_serving_capacity_model.md",
            "serving_calibration": {
                "effective_compute_utilization": serving.effective_utilization,
                "effective_memory_utilization": serving.effective_memory_utilization,
                "interconnect_utilization": serving.interconnect_utilization,
            },
            "token_distribution": token_distribution.to_dict(),
            "parallel_layout": layout.to_dict(),
            "data_parallel_replica_semantics": {
                "dp_replica_model": layout.dp_replica_model,
                "dp": layout.dp,
                "dp_attention": layout.dp_attention,
                "dp_expert": layout.dp_expert,
                "dp_replica_factor": layout.dp_replica_factor,
                "attention_replicas_total": layout.dp_attention * deployment_group_count,
                "expert_replicas_total": layout.dp_expert * deployment_group_count,
            },
            "pd": {
                **pool_plan,
                "mode": serving.pd_mode,
                "colocated_request_throughput_formula": "1 / (S_p50 / prefill_tokens_per_s + O_p50 / decode_tokens_per_s)",
                "disaggregated_request_throughput_formula": "min(prefill_pool_request_throughput_per_s, decode_pool_request_throughput_per_s, kv_handoff_request_limit_per_s)",
                "weight_replication": (
                    "prefill and decode pools each place a full deployment"
                    if serving.pd_mode == "disaggregated"
                    else "prefill and decode share one colocated deployment pool"
                ),
            },
            "isl": communication_summary,
            "weight_memory_breakdown_gb_per_satellite": weight_breakdown,
            "single_satellite_multi_card_parallelism": "not_modeled_todo",
            "moe_routing": {
                "moe_imbalance_factor": parallelism.moe_imbalance_factor,
                "alpha_ep": parallelism.alpha_ep,
                "assumption": (
                    "v1 assumes balanced MoE routing across experts; hot expert imbalance "
                    "and unique expert working-set expansion are not modeled"
                ),
            },
            "pipeline_bubble": {
                "prefill": "applies configured bubble_multiplier to throughput processing time",
                "decode": "continuous batching approximation; pipeline bubble is not applied to decode throughput processing time",
            },
            "prefill_stage": prefill,
            "decode_stage": decode,
            "kv_handoff": kv_handoff,
        },
    )






def _estimate_prefill(
    serving: ServingSpec,
    model: LLMModelProfile,
    hardware: SatelliteHardwareProfile,
    parallelism: ServingParallelism,
    layout: ParallelLayout,
    *,
    prompt_tokens: int,
    replicas_per_group: int,
    deployment_group_count: int,
    pd_handoff_active: bool,
) -> dict[str, Any]:
    prefill_batch_size = serving.prefill.batch_size
    if serving.prefill.token_throughput_per_s_per_accelerator is not None:
        interconnect = _interconnect_profile(
            serving,
            model,
            hardware,
            parallelism,
            token_count=prefill_batch_size * prompt_tokens,
            pd_handoff_active=pd_handoff_active,
        )
        if _manual_override_is_unsupported(parallelism):
            return {
                **interconnect,
                "latency_s": math.inf,
                "throughput_processing_time_s": math.inf,
                "prefill_batch_service_time_s": math.inf,
                "token_throughput_per_s": None,
                "interconnect_time_s": interconnect["total_time_s"],
                "bottleneck": "manual_prefill_throughput_unsupported_for_cross_satellite_parallelism",
                "warnings": (
                    "prefill manual throughput override is only supported for single-satellite/no-stage-DP runs; "
                    "use roofline fields for TP/PP/EP/DP throughput",
                    *interconnect["warnings"],
                ),
            }
        latency_throughput, serving_throughput = _manual_stage_throughputs(
            serving.prefill,
            hardware,
            parallelism,
            layout,
            model=model,
            replicas_per_group=replicas_per_group,
            deployment_group_count=deployment_group_count,
        )
        interconnect_s = interconnect["total_time_s"]
        latency_s = prefill_batch_size * prompt_tokens / latency_throughput + interconnect_s
        throughput_processing_time_s = prefill_batch_size * prompt_tokens / max(1e-12, serving_throughput) + interconnect_s
        return {
            **interconnect,
            "latency_s": latency_s,
            "throughput_processing_time_s": throughput_processing_time_s,
            "prefill_batch_service_time_s": throughput_processing_time_s,
            "token_throughput_per_s": _stage_throughput(
                prefill_batch_size * prompt_tokens,
                throughput_processing_time_s,
                replicas_per_group=replicas_per_group,
                deployment_group_count=deployment_group_count,
            ),
            "interconnect_time_s": interconnect_s,
            "bottleneck": "interconnect" if interconnect_s > 0 else "manual_prefill_throughput",
            "warnings": (
                "prefill throughput uses configured manual override for compute/memory only",
                *interconnect["warnings"],
            ),
        }

    flops = model.prefill_flop_breakdown(prompt_tokens)
    non_routed_bytes = model.non_routed_memory_gb * BYTES_PER_GB + (
        prefill_batch_size * prompt_tokens * model.kv_cache_bytes_per_token
    )
    expert_bytes = model.active_routed_memory_gb * BYTES_PER_GB
    stage_times = _stage_roofline_times(
        serving,
        hardware,
        parallelism,
        layout,
        model=model,
        non_routed_flops=prefill_batch_size * flops["non_routed"],
        routed_expert_flops=prefill_batch_size * flops["routed_expert"],
        non_routed_bytes=non_routed_bytes,
        routed_expert_bytes=expert_bytes,
    )
    interconnect = _interconnect_profile(
        serving,
        model,
        hardware,
        parallelism,
        token_count=prefill_batch_size * prompt_tokens,
        pd_handoff_active=pd_handoff_active,
    )
    interconnect_s = interconnect["total_time_s"]
    latency_s = stage_times["latency_s"] + interconnect_s
    throughput_processing_time_s = stage_times["stage_throughput_processing_time_s"] * parallelism.bubble_multiplier + interconnect_s
    throughput = _stage_throughput(
        prefill_batch_size * prompt_tokens,
        throughput_processing_time_s,
        replicas_per_group=replicas_per_group,
        deployment_group_count=deployment_group_count,
    )
    return {
        **interconnect,
        **stage_times,
        "latency_s": latency_s,
        "throughput_processing_time_s": throughput_processing_time_s,
        "prefill_batch_service_time_s": throughput_processing_time_s,
        "token_throughput_per_s": throughput,
        "interconnect_time_s": interconnect_s,
        "bottleneck": _bottleneck_from_parts(stage_times, interconnect_s),
        "warnings": interconnect["warnings"],
    }


def _estimate_decode(
    serving: ServingSpec,
    model: LLMModelProfile,
    hardware: SatelliteHardwareProfile,
    parallelism: ServingParallelism,
    layout: ParallelLayout,
    *,
    context_tokens: int,
    replicas_per_group: int,
    deployment_group_count: int,
    pd_handoff_active: bool,
) -> dict[str, Any]:
    decode_batch_size = serving.decode.batch_size
    if serving.decode.token_throughput_per_s_per_accelerator is not None:
        interconnect = _interconnect_profile(
            serving,
            model,
            hardware,
            parallelism,
            token_count=decode_batch_size,
            pd_handoff_active=pd_handoff_active,
        )
        if _manual_override_is_unsupported(parallelism):
            return {
                **interconnect,
                "latency_s": math.inf,
                "throughput_processing_time_s": math.inf,
                "decode_batch_service_time_s": math.inf,
                "token_throughput_per_s": None,
                "interconnect_time_s": interconnect["total_time_s"],
                "bottleneck": "manual_decode_throughput_unsupported_for_cross_satellite_parallelism",
                "warnings": (
                    "decode manual throughput override is only supported for single-satellite/no-stage-DP runs; "
                    "use roofline fields for TP/PP/EP/DP throughput",
                    *interconnect["warnings"],
                ),
            }
        latency_throughput, serving_throughput = _manual_stage_throughputs(
            serving.decode,
            hardware,
            parallelism,
            layout,
            model=model,
            replicas_per_group=replicas_per_group,
            deployment_group_count=deployment_group_count,
        )
        interconnect_s = interconnect["total_time_s"]
        latency_s = decode_batch_size / latency_throughput + interconnect_s
        throughput_processing_time_s = decode_batch_size / max(1e-12, serving_throughput) + interconnect_s
        return {
            **interconnect,
            "latency_s": latency_s,
            "throughput_processing_time_s": throughput_processing_time_s,
            "decode_batch_service_time_s": throughput_processing_time_s,
            "token_throughput_per_s": _stage_throughput(
                decode_batch_size,
                throughput_processing_time_s,
                replicas_per_group=replicas_per_group,
                deployment_group_count=deployment_group_count,
            ),
            "interconnect_time_s": interconnect_s,
            "bottleneck": "interconnect" if interconnect_s > 0 else "manual_decode_throughput",
            "warnings": (
                "decode throughput uses configured manual override for compute/memory only",
                *interconnect["warnings"],
            ),
        }

    flops = model.decode_flop_breakdown_per_token(context_tokens)
    non_routed_bytes = model.non_routed_memory_gb * BYTES_PER_GB + (
        decode_batch_size * context_tokens * model.kv_cache_bytes_per_token
    )
    expert_bytes = model.active_routed_memory_gb * BYTES_PER_GB
    stage_times = _stage_roofline_times(
        serving,
        hardware,
        parallelism,
        layout,
        model=model,
        non_routed_flops=decode_batch_size * flops["non_routed"],
        routed_expert_flops=decode_batch_size * flops["routed_expert"],
        non_routed_bytes=non_routed_bytes,
        routed_expert_bytes=expert_bytes,
    )
    interconnect = _interconnect_profile(
        serving,
        model,
        hardware,
        parallelism,
        token_count=decode_batch_size,
        pd_handoff_active=pd_handoff_active,
    )
    interconnect_s = interconnect["total_time_s"]
    latency_s = stage_times["latency_s"] + interconnect_s
    throughput_processing_time_s = stage_times["stage_throughput_processing_time_s"] + interconnect_s
    throughput = _stage_throughput(
        decode_batch_size,
        throughput_processing_time_s,
        replicas_per_group=replicas_per_group,
        deployment_group_count=deployment_group_count,
    )
    return {
        **interconnect,
        **stage_times,
        "latency_s": latency_s,
        "throughput_processing_time_s": throughput_processing_time_s,
        "decode_batch_service_time_s": throughput_processing_time_s,
        "token_throughput_per_s": throughput,
        "interconnect_time_s": interconnect_s,
        "bottleneck": _bottleneck_from_parts(stage_times, interconnect_s),
        "warnings": interconnect["warnings"],
    }


def _resolve_parallelism(
    requested: ServingParallelism,
    model: LLMModelProfile,
    hardware: SatelliteHardwareProfile,
    *,
    constellation_satellite_count: int | None,
    memory_reserve_fraction: float,
) -> tuple[ServingParallelism, tuple[str, ...]]:
    if not requested.auto_select:
        return requested, ()
    max_satellites = requested.max_satellites
    if constellation_satellite_count is not None:
        max_satellites = min(max_satellites, constellation_satellite_count)
    usable_memory_per_sat = hardware.total_fast_mem_gb * (1.0 - memory_reserve_fraction)
    warnings: list[str] = []
    if requested.strategy in ("auto_hybrid", "auto", "auto_expert"):
        for candidate in _iter_parallel_candidates(max_satellites, model=model, strategy=requested.strategy):
            breakdown = _weight_memory_breakdown_gb(model, candidate)
            if breakdown["total"] <= usable_memory_per_sat:
                warnings.append(
                    "auto parallelism selected "
                    f"TP={candidate.tensor_parallel_degree}, PP={candidate.pipeline_parallel_degree}, "
                    f"EP={candidate.expert_parallel_degree}"
                )
                return replace(
                    requested,
                    strategy="auto_selected",
                    tensor_parallel_degree=candidate.tensor_parallel_degree,
                    pipeline_parallel_degree=candidate.pipeline_parallel_degree,
                    expert_parallel_degree=candidate.expert_parallel_degree,
                    max_satellites=max_satellites,
                ), tuple(warnings)
        warnings.append("auto parallelism could not fit model weights within max_satellites")
        return replace(requested, max_satellites=max_satellites), tuple(warnings)

    for sat_count in range(1, max_satellites + 1):
        tp = sat_count
        pp = 1
        if requested.strategy in ("auto_pipeline", "pipeline_parallel"):
            tp = 1
            pp = sat_count
        candidate = replace(
            requested,
            tensor_parallel_degree=tp,
            pipeline_parallel_degree=pp,
            expert_parallel_degree=1,
        )
        if _weight_memory_breakdown_gb(model, candidate)["total"] <= usable_memory_per_sat:
            warnings.append(f"auto parallelism selected TP={tp}, PP={pp}")
            return replace(
                requested,
                strategy="auto_selected",
                tensor_parallel_degree=tp,
                pipeline_parallel_degree=pp,
                max_satellites=max_satellites,
            ), tuple(warnings)
    warnings.append("auto parallelism could not fit model weights within max_satellites")
    return replace(requested, max_satellites=max_satellites), tuple(warnings)


def _iter_parallel_candidates(
    max_satellites: int,
    *,
    model: LLMModelProfile,
    strategy: str,
) -> list[ServingParallelism]:
    candidates: list[ServingParallelism] = []
    max_tp = min(4, max_satellites)
    max_ep = min(model.expert_count or max_satellites, max_satellites)
    for tp in range(1, max_tp + 1):
        for pp in range(1, max_satellites + 1):
            for ep in range(1, max_ep + 1):
                if strategy == "auto_expert" and (tp != 1 or pp != 1):
                    continue
                if strategy == "auto" and model.is_moe and tp != 1:
                    continue
                if not model.is_moe and ep != 1:
                    continue
                candidate = ServingParallelism(
                    strategy="candidate",
                    tensor_parallel_degree=tp,
                    pipeline_parallel_degree=pp,
                    expert_parallel_degree=ep,
                    max_satellites=max_satellites,
                )
                if candidate.satellite_count <= max_satellites:
                    candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda item: (
            item.satellite_count,
            item.pipeline_parallel_degree,
            item.expert_parallel_degree,
            item.tensor_parallel_degree,
        ),
    )


def _interconnect_profile(
    serving: ServingSpec,
    model: LLMModelProfile,
    hardware: SatelliteHardwareProfile,
    parallelism: ServingParallelism,
    *,
    token_count: int,
    pd_handoff_active: bool = False,
    kv_handoff_bytes: float = 0.0,
    extra_required_flows: int = 0,
) -> dict[str, Any]:
    bytes_by_kind = {
        "tp": 0.0,
        "ep": 0.0,
        "pp": 0.0,
        "kv_handoff": kv_handoff_bytes,
    }
    flows_by_kind = {
        "tp": 0,
        "ep": 0,
        "pp": 0,
        "kv_handoff": 1 if pd_handoff_active or kv_handoff_bytes > 0 else 0,
    }
    if parallelism.tensor_parallel_degree > 1:
        bytes_by_kind["tp"] = _tensor_parallel_bytes(
            model,
            token_count=token_count,
            tensor_degree=parallelism.tensor_parallel_degree,
            layer_divisor=parallelism.pipeline_parallel_degree,
            collectives_per_layer=parallelism.tensor_parallel_collectives_per_layer,
        )
        flows_by_kind["tp"] = parallelism.tensor_parallel_degree
    if parallelism.pipeline_parallel_degree > 1:
        bytes_by_kind["pp"] = (
            (parallelism.pipeline_parallel_degree - 1)
            * token_count
            * (model.hidden_size or 0)
            * model.precision.activation_bytes
        )
        flows_by_kind["pp"] = parallelism.pipeline_parallel_degree - 1
    if parallelism.expert_parallel_degree > 1 and model.is_moe:
        bytes_by_kind["ep"] = _expert_parallel_bytes(
            model,
            token_count=token_count,
            parallelism=parallelism,
        )
        flows_by_kind["ep"] = parallelism.expert_parallel_degree

    total_bytes = sum(bytes_by_kind.values())
    concurrent_stage_flows = max(0, int(extra_required_flows))
    required_flows = sum(flows_by_kind.values()) + concurrent_stage_flows
    isl = _isl_bandwidth_profile(serving, hardware, required_flows)
    warnings = list(isl["warnings"])
    bytes_per_s = (isl["effective_bandwidth_gbps"] or 0.0) / 8.0 * 1e9
    times = {
        "tp_communication_time_s": _safe_ratio(bytes_by_kind["tp"], bytes_per_s) if bytes_by_kind["tp"] > 0 else 0.0,
        "ep_communication_time_s": _safe_ratio(bytes_by_kind["ep"], bytes_per_s) if bytes_by_kind["ep"] > 0 else 0.0,
        "pp_communication_time_s": _safe_ratio(bytes_by_kind["pp"], bytes_per_s) if bytes_by_kind["pp"] > 0 else 0.0,
        "kv_handoff_time_s": (
            _safe_ratio(bytes_by_kind["kv_handoff"], bytes_per_s)
            if bytes_by_kind["kv_handoff"] > 0
            else 0.0
        ),
    }
    if total_bytes > 0 and not math.isfinite(sum(times.values())):
        warnings.append("cross-satellite TP/PP/EP/PD communication requires positive ISL bandwidth")
    startup_s = _expert_parallel_startup_s(model, parallelism)
    times["ep_communication_time_s"] += startup_s
    total_time_s = sum(times.values())
    bottleneck = _communication_bottleneck(times)
    if isl["head_occupancy"] is not None and isl["head_occupancy"] > 1.0:
        warnings.append("ISL laser head contention")
    return {
        "total_time_s": total_time_s,
        "interconnect_time_s": total_time_s,
        "tp_communication_bytes": bytes_by_kind["tp"],
        "ep_communication_bytes": bytes_by_kind["ep"],
        "pp_communication_bytes": bytes_by_kind["pp"],
        "kv_handoff_bytes": bytes_by_kind["kv_handoff"],
        "flows_by_kind": {
            **flows_by_kind,
            "concurrent_stage": concurrent_stage_flows,
        },
        "tp_communication_time_s": times["tp_communication_time_s"],
        "ep_communication_time_s": times["ep_communication_time_s"],
        "pp_communication_time_s": times["pp_communication_time_s"],
        "kv_handoff_time_s": times["kv_handoff_time_s"],
        "effective_isl_bandwidth_gbps": isl["effective_bandwidth_gbps"],
        "isl_required_flows": required_flows,
        "available_isl_laser_heads": isl["available_laser_heads"],
        "isl_head_occupancy": isl["head_occupancy"],
        "communication_bottleneck": bottleneck,
        "communication_bottleneck_time_s": times.get(bottleneck, 0.0) if bottleneck else 0.0,
        "warnings": tuple(dict.fromkeys(warnings)),
    }


def _isl_bandwidth_profile(
    serving: ServingSpec,
    hardware: SatelliteHardwareProfile,
    required_flows: int,
) -> dict[str, Any]:
    available_heads = max(0, int(hardware.laser_terminal_count))
    terminal_bw = hardware.resolved_laser_terminal_bandwidth_gbps
    bandwidth_cap = hardware.inter_satellite_bandwidth_gbps * serving.interconnect_utilization
    base_bandwidth_gbps = min(
        bandwidth_cap,
        hardware.laser_terminal_count * terminal_bw,
    )
    if required_flows <= 0:
        head_occupancy = 0.0
    elif available_heads <= 0:
        head_occupancy = math.inf
    else:
        head_occupancy = required_flows / available_heads
    warnings: list[str] = []
    if required_flows > 0 and available_heads <= 0:
        warnings.append("ISL communication requires at least one installed laser terminal")
    if required_flows > 0 and base_bandwidth_gbps <= 0:
        warnings.append("ISL communication requires positive installed bandwidth")
    return {
        "effective_bandwidth_gbps": base_bandwidth_gbps,
        "available_laser_heads": available_heads,
        "head_occupancy": head_occupancy,
        "base_bandwidth_gbps": base_bandwidth_gbps,
        "bandwidth_cap_gbps": bandwidth_cap,
        "laser_head_cap_gbps": hardware.laser_terminal_count * terminal_bw,
        "laser_terminal_bandwidth_gbps": terminal_bw,
        "laser_terminal_bandwidth_source": (
            "configured"
            if hardware.laser_terminal_bandwidth_gbps is not None
            else "derived_from_total_installed_bandwidth"
        ),
        "warnings": tuple(warnings),
    }


def _communication_bottleneck(times: dict[str, float]) -> str | None:
    positive = {key: value for key, value in times.items() if value > 0}
    if not positive:
        return None
    return max(positive, key=positive.get)


def _kv_handoff_profile(
    serving: ServingSpec,
    model: LLMModelProfile,
    hardware: SatelliteHardwareProfile,
    parallelism: ServingParallelism,
    *,
    prompt_tokens_latency: int,
    prompt_token_count_for_throughput: int,
    base_required_flows: int = 0,
) -> dict[str, Any]:
    if serving.pd_mode == "colocated":
        return {
            "latency_s": 0.0,
            "request_throughput_per_s": None,
            "kv_handoff_request_limit_per_s": None,
            "bottleneck": "none",
            "kv_handoff_bytes_per_request_latency": 0.0,
            "kv_handoff_bytes_per_request_for_throughput": 0.0,
            "total_time_s": 0.0,
            "tp_communication_time_s": 0.0,
            "ep_communication_time_s": 0.0,
            "pp_communication_time_s": 0.0,
            "kv_handoff_time_s": 0.0,
            "effective_isl_bandwidth_gbps": hardware.inter_satellite_bandwidth_gbps * serving.interconnect_utilization,
            "isl_required_flows": base_required_flows,
            "flows_by_kind": {
                "tp": 0,
                "ep": 0,
                "pp": 0,
                "kv_handoff": 0,
                "concurrent_stage": base_required_flows,
            },
            "available_isl_laser_heads": hardware.laser_terminal_count,
            "isl_head_occupancy": 0.0,
            "communication_bottleneck": None,
            "warnings": (),
        }

    latency_bytes = prompt_tokens_latency * model.kv_cache_bytes_per_token
    throughput_bytes_per_request = prompt_token_count_for_throughput * model.kv_cache_bytes_per_token
    profile = _interconnect_profile(
        serving,
        model,
        hardware,
        parallelism,
        token_count=0,
        pd_handoff_active=True,
        kv_handoff_bytes=latency_bytes,
        extra_required_flows=base_required_flows,
    )
    request_throughput = None
    if throughput_bytes_per_request > 0 and profile["effective_isl_bandwidth_gbps"] is not None:
        bytes_per_s = profile["effective_isl_bandwidth_gbps"] / 8.0 * 1e9
        request_throughput = _safe_ratio(bytes_per_s, throughput_bytes_per_request)
        if not math.isfinite(request_throughput):
            request_throughput = None
    warnings = list(profile["warnings"])
    if latency_bytes > 0 and not math.isfinite(profile["kv_handoff_time_s"]):
        warnings.append("pd_mode=disaggregated requires effective ISL bandwidth for KV handoff")
    return {
        **profile,
        "latency_s": profile["kv_handoff_time_s"],
        "request_throughput_per_s": request_throughput,
        "kv_handoff_request_limit_per_s": request_throughput,
        "bottleneck": "kv_handoff" if profile["kv_handoff_time_s"] > 0 else "none",
        "kv_handoff_bytes_per_request_latency": latency_bytes,
        "kv_handoff_bytes_per_request_for_throughput": throughput_bytes_per_request,
        "warnings": tuple(dict.fromkeys(warnings)),
    }


def _merge_communication_summaries(
    prefill: dict[str, Any],
    decode: dict[str, Any],
    kv_handoff: dict[str, Any],
) -> dict[str, Any]:
    profiles = (prefill, decode, kv_handoff)
    time_keys = ("tp_communication_time_s", "ep_communication_time_s", "pp_communication_time_s", "kv_handoff_time_s")
    times = {
        key: max(float(profile.get(key, 0.0) or 0.0) for profile in profiles)
        for key in time_keys
    }
    bandwidths = [
        profile.get("effective_isl_bandwidth_gbps")
        for profile in profiles
        if profile.get("effective_isl_bandwidth_gbps") is not None
    ]
    occupancies = [
        profile.get("isl_head_occupancy")
        for profile in profiles
        if profile.get("isl_head_occupancy") is not None
    ]
    bottleneck = _communication_bottleneck(times)
    return {
        "effective_isl_bandwidth_gbps": min(bandwidths) if bandwidths else None,
        "isl_required_flows": max(int(profile.get("isl_required_flows", 0) or 0) for profile in profiles),
        "available_isl_laser_heads": min(
            int(profile.get("available_isl_laser_heads", 0) or 0) for profile in profiles
        ),
        "flows_by_kind": {
            "prefill": prefill.get("flows_by_kind", {}),
            "decode": decode.get("flows_by_kind", {}),
            "kv_handoff": kv_handoff.get("flows_by_kind", {}),
        },
        "isl_head_occupancy": max(occupancies) if occupancies else None,
        "tp_communication_time_s": times["tp_communication_time_s"],
        "ep_communication_time_s": times["ep_communication_time_s"],
        "pp_communication_time_s": times["pp_communication_time_s"],
        "kv_handoff_time_s": times["kv_handoff_time_s"],
        "communication_bottleneck": bottleneck,
        "communication_bottleneck_time_s": times.get(bottleneck, 0.0) if bottleneck else 0.0,
    }


def _tensor_parallel_bytes(
    model: LLMModelProfile,
    *,
    token_count: int,
    tensor_degree: int,
    layer_divisor: int,
    collectives_per_layer: int,
) -> float:
    if tensor_degree <= 1 or collectives_per_layer <= 0:
        return 0.0
    allreduce_factor = 2.0 * (tensor_degree - 1) / tensor_degree
    effective_layers = (model.layer_count or 0) / max(1, layer_divisor)
    return (
        token_count
        * collectives_per_layer
        * effective_layers
        * (model.hidden_size or 0)
        * model.precision.activation_bytes
        * allreduce_factor
    )


def _expert_parallel_bytes(
    model: LLMModelProfile,
    *,
    token_count: int,
    parallelism: ServingParallelism,
) -> float:
    ep = parallelism.expert_parallel_degree
    if ep <= 1 or not model.is_moe:
        return 0.0
    remote_fraction = (ep - 1) / ep
    moe_layers = model.resolved_moe_layer_count / max(1, parallelism.pipeline_parallel_degree)
    return (
        token_count
        * 2.0
        * (model.active_expert_count or 1)
        * (model.hidden_size or 0)
        * model.precision.activation_bytes
        * moe_layers
        * remote_fraction
        * parallelism.expert_parallel_imbalance_factor
    )


def _expert_parallel_startup_s(model: LLMModelProfile, parallelism: ServingParallelism) -> float:
    ep = parallelism.expert_parallel_degree
    if ep <= 1 or not model.is_moe or parallelism.expert_parallel_startup_us <= 0:
        return 0.0
    moe_layers = model.resolved_moe_layer_count / max(1, parallelism.pipeline_parallel_degree)
    return parallelism.expert_parallel_startup_us * 1e-6 * moe_layers * math.log2(ep)


def _deployment_group_count(
    satellite_count_used: int,
    constellation_satellite_count: int | None,
) -> int:
    if constellation_satellite_count is None:
        return 1
    if satellite_count_used <= 0:
        return 0
    return max(0, constellation_satellite_count // satellite_count_used)


def _parallel_layout(model: LLMModelProfile, parallelism: ServingParallelism) -> ParallelLayout:
    tp = parallelism.tensor_parallel_degree
    pp = parallelism.pipeline_parallel_degree
    ep = parallelism.expert_parallel_degree
    dp = parallelism.data_parallel_degree if parallelism.stage_dp_mode != "none" else 1
    dp_attention = parallelism.effective_dp_attention
    dp_expert = parallelism.effective_dp_expert
    warnings: list[str] = []
    if parallelism.stage_dp_mode == "uniform":
        stage_satellite_count = dp * math.lcm(tp, ep)
        dp_replica_factor = dp
        dp_replica_model = "uniform"
    elif parallelism.stage_dp_mode == "path_specific":
        stage_satellite_count = math.lcm(tp * dp_attention, ep * dp_expert)
        dp_replica_factor = max(dp_attention, dp_expert)
        dp_replica_model = "path_specific"
    else:
        stage_satellite_count = math.lcm(tp, ep)
        dp_replica_factor = 1
        dp_replica_model = "none"
        if parallelism.data_parallel_degree > 1 or parallelism.dp_attention or parallelism.dp_expert:
            warnings.append("stage_dp_mode=none ignores dp, dp_attention, and dp_expert for throughput")

    has_routed_path = model.is_moe and model.active_routed_parameters_b > 0
    latency_parallelism_routed_path = _expert_latency_degree(model, parallelism) if has_routed_path else 0
    throughput_parallelism_routed_path = ep * dp_expert if has_routed_path else 0
    return ParallelLayout(
        stage_satellite_count=stage_satellite_count,
        satellites_per_deployment=pp * stage_satellite_count,
        dp=dp,
        dp_attention=dp_attention,
        dp_expert=dp_expert,
        dp_replica_factor=dp_replica_factor,
        dp_replica_model=dp_replica_model,
        latency_parallelism_always_path=tp,
        latency_parallelism_routed_path=latency_parallelism_routed_path,
        throughput_parallelism_always_path=tp * dp_attention,
        throughput_parallelism_routed_path=throughput_parallelism_routed_path,
        placement_warnings=tuple(warnings),
    )


def _parallelism_model_warnings(model: LLMModelProfile, parallelism: ServingParallelism) -> list[str]:
    warnings: list[str] = []
    pp = parallelism.pipeline_parallel_degree
    ep = parallelism.expert_parallel_degree
    if parallelism.stage_dp_mode == "uniform" and (
        parallelism.dp_attention is not None or parallelism.dp_expert is not None
    ):
        warnings.append("stage_dp_mode=uniform ignores dp_attention/dp_expert and uses dp for both paths")
    if parallelism.stage_dp_mode == "path_specific" and (
        parallelism.dp_attention is None or parallelism.dp_expert is None
    ):
        warnings.append("stage_dp_mode=path_specific defaults missing dp_attention/dp_expert to dp")
    if not model.is_moe and ep > 1:
        warnings.append("dense/non-MoE model ignores expert_parallel_degree for routed path throughput")
    if model.is_moe and model.expert_count is not None:
        if ep > model.expert_count:
            warnings.append(f"expert_parallel_degree {ep} exceeds expert_count {model.expert_count}")
        elif model.expert_count % ep != 0:
            warnings.append(
                "expert_count is not divisible by expert_parallel_degree; "
                "using ceil experts per rank for routed weight memory"
            )
    if pp > 1:
        if model.layer_count and model.layer_count % pp != 0:
            warnings.append("layer_count is not divisible by pipeline_parallel_degree; using ceil stage size")
        if model.is_moe and model.resolved_moe_layer_count and model.resolved_moe_layer_count % pp != 0:
            warnings.append("moe_layer_count is not divisible by pipeline_parallel_degree; using ceil stage size")
    if parallelism.moe_imbalance_factor != 1.0 or parallelism.alpha_ep != 0.0:
        warnings.append(
            "MoE imbalance parameters are recorded but v1 assumes balanced routing across experts"
        )
    return warnings


def _pd_pool_plan(
    serving: ServingSpec,
    *,
    satellites_per_deployment: int,
    constellation_satellite_count: int | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    satellite_budget = _normalised_satellite_budget(serving.satellite_budget)
    satellite_budget_mode = _satellite_budget_mode(satellite_budget)
    comparison_group_id = _optional_str(satellite_budget.get("comparison_group_id")) if satellite_budget else None
    serving_budget_count, satellite_budget_source, budget_warnings = _resolve_serving_satellite_budget_count(
        satellite_budget,
        constellation_satellite_count=constellation_satellite_count,
    )
    warnings.extend(budget_warnings)

    if serving.pd_mode == "colocated":
        pool_satellites = serving_budget_count if serving_budget_count is not None else constellation_satellite_count
        groups = _deployment_group_count(satellites_per_deployment, pool_satellites)
        return {
            "deployment_group_count": groups,
            "prefill_deployment_group_count": groups,
            "decode_deployment_group_count": groups,
            "satellite_count_used": satellites_per_deployment,
            "serving_satellite_budget_count": pool_satellites,
            "constellation_satellite_count": constellation_satellite_count,
            "satellite_budget_mode": satellite_budget_mode,
            "satellite_budget_source": satellite_budget_source,
            "comparison_group_id": comparison_group_id,
            "satellite_budget_unused_count": _satellite_budget_unused_count(
                constellation_satellite_count,
                pool_satellites,
            ),
            "prefill_pool_satellites": pool_satellites,
            "decode_pool_satellites": pool_satellites,
            "warnings": tuple(dict.fromkeys(warnings)),
        }

    prefill_fraction, decode_fraction, fraction_warnings, fractions_valid = _resolved_pd_satellite_fractions(
        pd_mode=serving.pd_mode,
        prefill_fraction=serving.prefill_satellite_fraction,
        decode_fraction=serving.decode_satellite_fraction,
    )
    warnings.extend(fraction_warnings)
    if not fractions_valid:
        return {
            "deployment_group_count": 0,
            "prefill_deployment_group_count": 0,
            "decode_deployment_group_count": 0,
            "satellite_count_used": 0,
            "serving_satellite_budget_count": serving_budget_count,
            "constellation_satellite_count": constellation_satellite_count,
            "satellite_budget_mode": satellite_budget_mode,
            "satellite_budget_source": satellite_budget_source,
            "comparison_group_id": comparison_group_id,
            "satellite_budget_unused_count": _satellite_budget_unused_count(
                constellation_satellite_count,
                serving_budget_count,
            ),
            "prefill_pool_satellites": 0,
            "decode_pool_satellites": 0,
            "prefill_satellite_fraction": prefill_fraction,
            "decode_satellite_fraction": decode_fraction,
            "warnings": tuple(dict.fromkeys(warnings)),
        }

    if serving_budget_count is None:
        if serving.prefill_satellite_fraction is not None or serving.decode_satellite_fraction is not None:
            warnings.append("PD satellite fractions are ignored when constellation_satellite_count is not configured")
        if satellite_budget:
            warnings.append("satellite_budget is ignored when no active or constellation satellite count is configured")
        return {
            "deployment_group_count": 1,
            "prefill_deployment_group_count": 1,
            "decode_deployment_group_count": 1,
            "satellite_count_used": 2 * satellites_per_deployment,
            "serving_satellite_budget_count": None,
            "constellation_satellite_count": constellation_satellite_count,
            "satellite_budget_mode": satellite_budget_mode,
            "satellite_budget_source": satellite_budget_source,
            "comparison_group_id": comparison_group_id,
            "satellite_budget_unused_count": None,
            "prefill_pool_satellites": satellites_per_deployment,
            "decode_pool_satellites": satellites_per_deployment,
            "prefill_satellite_fraction": prefill_fraction,
            "decode_satellite_fraction": decode_fraction,
            "warnings": tuple(dict.fromkeys(warnings)),
        }

    role_plan = _pd_role_pool_plan(
        satellite_budget,
        serving_budget_count=serving_budget_count,
        prefill_fraction=prefill_fraction,
        decode_fraction=decode_fraction,
    )
    warnings.extend(role_plan["warnings"])
    if not role_plan["valid"]:
        return {
            "deployment_group_count": 0,
            "prefill_deployment_group_count": 0,
            "decode_deployment_group_count": 0,
            "satellite_count_used": 0,
            "serving_satellite_budget_count": serving_budget_count,
            "constellation_satellite_count": constellation_satellite_count,
            "satellite_budget_mode": satellite_budget_mode,
            "satellite_budget_source": satellite_budget_source,
            "comparison_group_id": comparison_group_id,
            "satellite_budget_unused_count": _satellite_budget_unused_count(
                constellation_satellite_count,
                serving_budget_count,
            ),
            "prefill_pool_satellites": 0,
            "decode_pool_satellites": 0,
            "prefill_satellite_fraction": prefill_fraction,
            "decode_satellite_fraction": decode_fraction,
            "warnings": tuple(dict.fromkeys(warnings)),
        }

    prefill_pool_satellites = role_plan["prefill_pool_satellites"]
    decode_pool_satellites = role_plan["decode_pool_satellites"]
    if serving_budget_count > 0 and "fraction" not in str(role_plan["source"]):
        prefill_fraction = prefill_pool_satellites / serving_budget_count
        decode_fraction = decode_pool_satellites / serving_budget_count
    prefill_groups = prefill_pool_satellites // max(1, satellites_per_deployment)
    decode_groups = decode_pool_satellites // max(1, satellites_per_deployment)
    paired_groups = min(prefill_groups, decode_groups)
    return {
        "deployment_group_count": paired_groups,
        "prefill_deployment_group_count": prefill_groups,
        "decode_deployment_group_count": decode_groups,
        "satellite_count_used": min(serving_budget_count, prefill_pool_satellites + decode_pool_satellites),
        "serving_satellite_budget_count": serving_budget_count,
        "constellation_satellite_count": constellation_satellite_count,
        "satellite_budget_mode": satellite_budget_mode,
        "satellite_budget_source": satellite_budget_source,
        "pd_role_budget_source": role_plan["source"],
        "comparison_group_id": comparison_group_id,
        "satellite_budget_unused_count": _satellite_budget_unused_count(
            constellation_satellite_count,
            serving_budget_count,
        ),
        "role_budget_unused_count": max(0, serving_budget_count - prefill_pool_satellites - decode_pool_satellites),
        "prefill_pool_satellites": prefill_pool_satellites,
        "decode_pool_satellites": decode_pool_satellites,
        "prefill_satellite_fraction": prefill_fraction,
        "decode_satellite_fraction": decode_fraction,
        "warnings": tuple(dict.fromkeys(warnings)),
    }


def _normalised_satellite_budget(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _satellite_budget_mode(satellite_budget: dict[str, Any]) -> str:
    if not satellite_budget:
        return "full_constellation_default"
    return str(satellite_budget.get("mode") or "fixed_total").lower()


def _resolve_serving_satellite_budget_count(
    satellite_budget: dict[str, Any],
    *,
    constellation_satellite_count: int | None,
) -> tuple[int | None, str, tuple[str, ...]]:
    warnings_out: list[str] = []
    if not satellite_budget:
        return constellation_satellite_count, "full_constellation_default", ()

    count, count_key = _budget_int_from_keys(
        satellite_budget,
        (
            "active_satellite_count",
            "active_satellites",
            "serving_satellite_budget_count",
            "serving_satellites",
            "satellite_count",
            "total_satellites",
        ),
    )
    if count is not None:
        resolved = max(0, count)
        source = count_key or "active_satellite_count"
    else:
        fraction, fraction_key = _budget_float_from_keys(
            satellite_budget,
            (
                "active_satellite_fraction",
                "serving_satellite_fraction",
                "satellite_fraction",
                "total_fraction",
            ),
        )
        if fraction is not None:
            source = fraction_key or "active_satellite_fraction"
            if constellation_satellite_count is None:
                warnings_out.append("satellite_budget fraction requires constellation_satellite_count")
                return None, source, tuple(warnings_out)
            if not math.isfinite(fraction) or fraction < 0:
                warnings_out.append("satellite_budget fraction must be finite and non-negative")
                return 0, source, tuple(warnings_out)
            resolved = math.floor(constellation_satellite_count * fraction)
        else:
            resolved = constellation_satellite_count if constellation_satellite_count is not None else None
            source = "satellite_budget_default_full_constellation"

    if resolved is not None and constellation_satellite_count is not None and resolved > constellation_satellite_count:
        warnings_out.append(
            f"satellite_budget requested {resolved} satellites but constellation has {constellation_satellite_count}; clipping"
        )
        resolved = constellation_satellite_count
    return resolved, source, tuple(warnings_out)


def _pd_role_pool_plan(
    satellite_budget: dict[str, Any],
    *,
    serving_budget_count: int,
    prefill_fraction: float,
    decode_fraction: float,
) -> dict[str, Any]:
    warnings_out: list[str] = []
    prefill_count, prefill_count_key = _budget_int_from_keys(
        satellite_budget,
        (
            "prefill_satellites",
            "prefill_satellite_count",
            "prefill_active_satellite_count",
            "prefill_pool_satellites",
        ),
    )
    decode_count, decode_count_key = _budget_int_from_keys(
        satellite_budget,
        (
            "decode_satellites",
            "decode_satellite_count",
            "decode_active_satellite_count",
            "decode_pool_satellites",
        ),
    )
    if prefill_count is not None or decode_count is not None:
        if prefill_count is None:
            prefill_count = max(0, serving_budget_count - max(0, decode_count or 0))
            prefill_count_key = "complement_of_decode_satellites"
        if decode_count is None:
            decode_count = max(0, serving_budget_count - max(0, prefill_count or 0))
            decode_count_key = "complement_of_prefill_satellites"
        prefill_pool = max(0, int(prefill_count))
        decode_pool = max(0, int(decode_count))
        source = f"{prefill_count_key}+{decode_count_key}"
        return _validate_pd_role_pools(
            prefill_pool,
            decode_pool,
            serving_budget_count=serving_budget_count,
            source=source,
            warnings_out=warnings_out,
        )

    budget_prefill_fraction, budget_prefill_key = _budget_float_from_keys(
        satellite_budget,
        ("prefill_fraction", "prefill_satellite_fraction"),
    )
    budget_decode_fraction, budget_decode_key = _budget_float_from_keys(
        satellite_budget,
        ("decode_fraction", "decode_satellite_fraction"),
    )
    if budget_prefill_fraction is not None or budget_decode_fraction is not None:
        resolved_prefill, resolved_decode, fraction_warnings, fractions_valid = _resolved_pd_satellite_fractions(
            pd_mode="disaggregated",
            prefill_fraction=budget_prefill_fraction,
            decode_fraction=budget_decode_fraction,
        )
        warnings_out.extend(fraction_warnings)
        if not fractions_valid:
            return {
                "valid": False,
                "prefill_pool_satellites": 0,
                "decode_pool_satellites": 0,
                "source": "satellite_budget_role_fraction",
                "warnings": tuple(dict.fromkeys(warnings_out)),
            }
        prefill_fraction = resolved_prefill
        decode_fraction = resolved_decode
        source = f"{budget_prefill_key or 'complement_prefill_fraction'}+{budget_decode_key or 'complement_decode_fraction'}"
    else:
        source = "serving_pd_satellite_fraction"

    prefill_pool = math.floor(serving_budget_count * max(0.0, prefill_fraction))
    decode_pool = math.floor(serving_budget_count * max(0.0, decode_fraction))
    return _validate_pd_role_pools(
        prefill_pool,
        decode_pool,
        serving_budget_count=serving_budget_count,
        source=source,
        warnings_out=warnings_out,
    )


def _validate_pd_role_pools(
    prefill_pool: int,
    decode_pool: int,
    *,
    serving_budget_count: int,
    source: str,
    warnings_out: list[str],
) -> dict[str, Any]:
    valid = True
    if prefill_pool + decode_pool > serving_budget_count:
        warnings_out.append(
            "prefill/decode satellite counts exceed serving satellite budget"
        )
        valid = False
    return {
        "valid": valid,
        "prefill_pool_satellites": prefill_pool if valid else 0,
        "decode_pool_satellites": decode_pool if valid else 0,
        "source": source,
        "warnings": tuple(dict.fromkeys(warnings_out)),
    }


def _budget_int_from_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> tuple[int | None, str | None]:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return int(value), key
    return None, None


def _budget_float_from_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> tuple[float | None, str | None]:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value), key
    return None, None


def _satellite_budget_unused_count(
    constellation_satellite_count: int | None,
    serving_budget_count: int | None,
) -> int | None:
    if constellation_satellite_count is None or serving_budget_count is None:
        return None
    return max(0, constellation_satellite_count - serving_budget_count)


def _resolved_pd_satellite_fractions(
    *,
    pd_mode: str,
    prefill_fraction: float | None,
    decode_fraction: float | None,
    fail_on_invalid: bool = False,
) -> tuple[float, float, tuple[str, ...], bool]:
    if pd_mode != "disaggregated":
        return (
            float(prefill_fraction) if prefill_fraction is not None else 0.0,
            float(decode_fraction) if decode_fraction is not None else 0.0,
            (),
            True,
        )

    if prefill_fraction is None and decode_fraction is None:
        resolved_prefill = 0.5
        resolved_decode = 0.5
    elif prefill_fraction is None:
        resolved_decode = float(decode_fraction or 0.0)
        resolved_prefill = 1.0 - resolved_decode
    elif decode_fraction is None:
        resolved_prefill = float(prefill_fraction)
        resolved_decode = 1.0 - resolved_prefill
    else:
        resolved_prefill = float(prefill_fraction)
        resolved_decode = float(decode_fraction)

    warnings_out: list[str] = []
    valid = True
    if not math.isfinite(resolved_prefill) or not math.isfinite(resolved_decode):
        warnings_out.append("pd_mode=disaggregated requires finite prefill/decode satellite fractions")
        valid = False
    if resolved_prefill <= 0 or resolved_decode <= 0:
        warnings_out.append("pd_mode=disaggregated requires positive prefill/decode satellite fractions")
        valid = False
    if resolved_prefill + resolved_decode > 1.0 + 1e-9:
        warnings_out.append("prefill/decode satellite fractions must sum to <= 1.0")
        valid = False
    if fail_on_invalid and not valid:
        raise ValueError("; ".join(warnings_out))
    return resolved_prefill, resolved_decode, tuple(warnings_out), valid


def _weight_memory_breakdown_gb(model: LLMModelProfile, parallelism: ServingParallelism) -> dict[str, float]:
    non_pipeline_fraction = _pipeline_stage_fraction(model.layer_count or 0, parallelism.pipeline_parallel_degree)
    non_routed = model.non_routed_memory_gb / parallelism.tensor_parallel_degree * non_pipeline_fraction
    routed = _routed_weight_memory_per_satellite_gb(model, parallelism)
    return {
        "non_routed": non_routed,
        "routed_expert": routed,
        "total": non_routed + routed,
        "dp_replica_note": "DP replicas do not reduce per-satellite shard memory",
    }


def _routed_weight_memory_per_satellite_gb(
    model: LLMModelProfile,
    parallelism: ServingParallelism,
) -> float:
    if not model.is_moe or model.routed_expert_memory_gb <= 0:
        return 0.0
    ep = parallelism.expert_parallel_degree
    pp = parallelism.pipeline_parallel_degree
    if model.expert_count and ep > 0:
        per_expert_gb = model.routed_expert_memory_gb / model.expert_count
        experts_per_ep_rank = math.ceil(model.expert_count / ep)
        routed_gb = per_expert_gb * experts_per_ep_rank
    else:
        routed_gb = model.routed_expert_memory_gb / ep
    routed_fraction = _pipeline_stage_fraction(model.resolved_moe_layer_count, pp)
    return routed_gb * routed_fraction


def _pipeline_stage_fraction(layer_count: int, pipeline_parallel_degree: int) -> float:
    if pipeline_parallel_degree <= 1 or layer_count <= 0:
        return 1.0
    return math.ceil(layer_count / pipeline_parallel_degree) / layer_count



def _expert_latency_degree(model: LLMModelProfile, parallelism: ServingParallelism) -> int:
    if not model.is_moe or model.active_routed_parameters_b <= 0:
        return 1
    active_experts = max(1, model.active_expert_count or 1)
    return max(1, min(active_experts, parallelism.expert_parallel_degree))


def _stage_roofline_times(
    serving: ServingSpec,
    hardware: SatelliteHardwareProfile,
    parallelism: ServingParallelism,
    layout: ParallelLayout,
    *,
    model: LLMModelProfile,
    non_routed_flops: float,
    routed_expert_flops: float,
    non_routed_bytes: float,
    routed_expert_bytes: float,
) -> dict[str, float]:
    compute_per_sat_flops_s = hardware.total_peak_compute_tflops * serving.effective_utilization * 1e12
    memory_per_sat_bytes_s = hardware.total_fast_mem_bw_gbs * serving.effective_memory_utilization * 1e9
    non_routed_latency_degree = max(1, layout.latency_parallelism_always_path)
    non_routed_throughput_parallelism = max(1, layout.throughput_parallelism_always_path)
    expert_latency_degree = max(1, layout.latency_parallelism_routed_path) if routed_expert_flops > 0 else 0
    expert_throughput_parallelism = max(1, layout.throughput_parallelism_routed_path) if routed_expert_flops > 0 else 0
    non_throughput_processing_scale = _pipeline_stage_fraction(model.layer_count or 0, parallelism.pipeline_parallel_degree)
    routed_throughput_processing_scale = _pipeline_stage_fraction(
        model.resolved_moe_layer_count,
        parallelism.pipeline_parallel_degree,
    )

    non_latency_compute_s = _safe_ratio(non_routed_flops, non_routed_latency_degree * compute_per_sat_flops_s)
    non_latency_memory_s = _safe_ratio(non_routed_bytes, non_routed_latency_degree * memory_per_sat_bytes_s)
    expert_latency_compute_s = (
        _safe_ratio(routed_expert_flops, expert_latency_degree * compute_per_sat_flops_s)
        if routed_expert_flops > 0
        else 0.0
    )
    expert_latency_memory_s = (
        _safe_ratio(routed_expert_bytes, expert_latency_degree * memory_per_sat_bytes_s)
        if routed_expert_flops > 0
        else 0.0
    )

    non_throughput_compute_time_s = _safe_ratio(
        non_routed_flops * non_throughput_processing_scale,
        non_routed_throughput_parallelism * compute_per_sat_flops_s,
    )
    non_throughput_memory_time_s = _safe_ratio(
        non_routed_bytes * non_throughput_processing_scale,
        non_routed_throughput_parallelism * memory_per_sat_bytes_s,
    )
    expert_throughput_compute_time_s = (
        _safe_ratio(
            routed_expert_flops * routed_throughput_processing_scale,
            expert_throughput_parallelism * compute_per_sat_flops_s,
        )
        if routed_expert_flops > 0
        else 0.0
    )
    expert_throughput_memory_time_s = (
        _safe_ratio(
            routed_expert_bytes * routed_throughput_processing_scale,
            expert_throughput_parallelism * memory_per_sat_bytes_s,
        )
        if routed_expert_flops > 0
        else 0.0
    )

    non_latency_s = max(non_latency_compute_s, non_latency_memory_s)
    expert_latency_s = max(expert_latency_compute_s, expert_latency_memory_s)
    non_throughput_processing_s = max(non_throughput_compute_time_s, non_throughput_memory_time_s)
    expert_throughput_processing_s = max(expert_throughput_compute_time_s, expert_throughput_memory_time_s)
    return {
        "latency_s": non_latency_s + expert_latency_s,
        "stage_throughput_processing_time_s": non_throughput_processing_s + expert_throughput_processing_s,
        "latency_compute_time_s": non_latency_compute_s + expert_latency_compute_s,
        "latency_memory_time_s": non_latency_memory_s + expert_latency_memory_s,
        "throughput_compute_time_s": non_throughput_compute_time_s + expert_throughput_compute_time_s,
        "throughput_memory_time_s": non_throughput_memory_time_s + expert_throughput_memory_time_s,
        "non_routed_latency_s": non_latency_s,
        "routed_expert_latency_s": expert_latency_s,
        "always_path_throughput_processing_time_s": non_throughput_processing_s,
        "routed_expert_path_throughput_processing_time_s": expert_throughput_processing_s,
        "non_routed_latency_degree": float(non_routed_latency_degree),
        "routed_expert_latency_degree": float(expert_latency_degree),
        "non_routed_throughput_parallelism": float(non_routed_throughput_parallelism),
        "routed_expert_throughput_parallelism": float(expert_throughput_parallelism),
        "stage_parallel_degree": float(layout.stage_satellite_count),
        "non_routed_pipeline_throughput_fraction": non_throughput_processing_scale,
        "routed_expert_pipeline_throughput_fraction": routed_throughput_processing_scale if routed_expert_flops > 0 else 0.0,
    }


def _stage_throughput(
    token_count: int,
    throughput_processing_time_s: float,
    *,
    replicas_per_group: int,
    deployment_group_count: int,
) -> float | None:
    replica_count = replicas_per_group * deployment_group_count
    if replica_count <= 0 or throughput_processing_time_s <= 0 or not math.isfinite(throughput_processing_time_s):
        return None
    return token_count / throughput_processing_time_s * replica_count


def _kv_cache_capacity_tokens(
    serving: ServingSpec,
    model: LLMModelProfile,
    parallelism: ServingParallelism,
    usable_memory_per_satellite_gb: float,
    weight_memory_per_satellite_gb: float,
) -> int | None:
    if serving.kv_cache_capacity_gb is not None:
        return math.floor(serving.kv_cache_capacity_gb * BYTES_PER_GB / max(1.0, model.kv_cache_bytes_per_token))
    kv_stage_fraction = _pipeline_stage_fraction(model.layer_count or 0, parallelism.pipeline_parallel_degree)
    kv_per_token_per_sat = (
        model.kv_cache_bytes_per_token
        / max(1, parallelism.tensor_parallel_degree)
        * kv_stage_fraction
    )
    if kv_per_token_per_sat <= 0:
        return None
    free_memory_gb = usable_memory_per_satellite_gb - weight_memory_per_satellite_gb
    if free_memory_gb <= 0:
        return 0
    return math.floor(free_memory_gb * BYTES_PER_GB / kv_per_token_per_sat)


def _manual_stage_throughputs(
    stage: StageServingProfile,
    hardware: SatelliteHardwareProfile,
    parallelism: ServingParallelism,
    layout: ParallelLayout,
    *,
    model: LLMModelProfile,
    replicas_per_group: int,
    deployment_group_count: int,
) -> tuple[float, float]:
    per_acc = stage.token_throughput_per_s_per_accelerator or 0.0
    latency_satellites = max(1, layout.latency_parallelism_always_path)
    active_throughput_parallelisms = [max(1, layout.throughput_parallelism_always_path)]
    if model.is_moe and model.active_routed_parameters_b > 0:
        active_throughput_parallelisms.append(max(1, layout.throughput_parallelism_routed_path))
    throughput_processing_satellites = min(active_throughput_parallelisms)
    latency_throughput = max(1e-12, per_acc * hardware.accelerator_count * latency_satellites)
    serving_throughput = per_acc * hardware.accelerator_count * throughput_processing_satellites
    return latency_throughput, max(1e-12, serving_throughput)


def _manual_override_is_unsupported(parallelism: ServingParallelism) -> bool:
    return parallelism.satellite_count > 1 or parallelism.stage_dp_mode != "none"


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _min_satellites_for_weights(model: LLMModelProfile, usable_memory_per_satellite_gb: float) -> int:
    if usable_memory_per_satellite_gb <= 0:
        return math.inf  # type: ignore[return-value]
    dense_bound = max(1, math.ceil(model.weight_memory_gb / usable_memory_per_satellite_gb))
    max_tp = max(1, min(64, math.ceil(model.non_routed_memory_gb / usable_memory_per_satellite_gb) + 1))
    max_ep = 1
    if model.is_moe:
        max_ep = max(1, min(model.expert_count or 1, math.ceil(model.routed_expert_memory_gb / usable_memory_per_satellite_gb) + 1))
    max_pp = max(1, min(64, dense_bound + 1))
    best: int | None = None
    for tp in range(1, max_tp + 1):
        for ep in range(1, max_ep + 1):
            for pp in range(1, max_pp + 1):
                parallelism = ServingParallelism(
                    tensor_parallel_degree=tp,
                    pipeline_parallel_degree=pp,
                    expert_parallel_degree=ep,
                )
                if best is not None and parallelism.satellite_count >= best:
                    continue
                if _weight_memory_breakdown_gb(model, parallelism)["total"] <= usable_memory_per_satellite_gb:
                    best = parallelism.satellite_count
    return best or dense_bound


def _bottleneck_from_parts(stage_times: dict[str, float], interconnect_s: float) -> str:
    values = {
        "non_routed": stage_times["non_routed_latency_s"],
        "routed_expert": stage_times["routed_expert_latency_s"],
        "interconnect": interconnect_s,
    }
    return max(values, key=values.get)


def _deployment_mode(parallelism: ServingParallelism) -> str:
    if parallelism.satellite_count <= 1:
        return "single_satellite"
    active = []
    if parallelism.tensor_parallel_degree > 1:
        active.append("tensor")
    if parallelism.pipeline_parallel_degree > 1:
        active.append("pipeline")
    if parallelism.expert_parallel_degree > 1:
        active.append("expert")
    if not active:
        return "single_satellite"
    return "_".join(active) + "_parallel"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.inf
    return numerator / denominator


def _unevaluated_throughput(message: str) -> ServingCapacityProfile:
    return ServingCapacityProfile(
        evaluated=False,
        feasible=False,
        deployment_mode="not_evaluated",
        pd_mode="colocated",
        model={},
        hardware={},
        parallelism={},
        stage_satellite_count=0,
        satellites_per_deployment=0,
        dp=0,
        dp_attention=0,
        dp_expert=0,
        dp_replica_factor=0,
        latency_parallelism_always_path=0,
        latency_parallelism_routed_path=0,
        throughput_parallelism_always_path=0,
        throughput_parallelism_routed_path=0,
        satellite_count_used=0,
        serving_satellite_budget_count=None,
        constellation_satellite_count=None,
        satellite_budget_mode=None,
        comparison_group_id=None,
        deployment_group_count=0,
        prefill_deployment_group_count=0,
        decode_deployment_group_count=0,
        local_tensor_parallel_degree=0,
        data_parallel_replicas_per_group=0,
        total_data_parallel_replicas=0,
        min_satellites_for_weights=0,
        weight_memory_gb=0.0,
        weight_memory_per_satellite_gb=0.0,
        usable_memory_per_satellite_gb=0.0,
        kv_cache_bytes_per_token=0.0,
        kv_cache_capacity_tokens=None,
        prefill_tokens_per_s=None,
        decode_tokens_per_s=None,
        request_throughput_per_s=None,
        ttft_s=None,
        e2e_s=None,
        prefill_interconnect_time_s=None,
        decode_interconnect_time_s=None,
        effective_isl_bandwidth_gbps=None,
        isl_required_flows=0,
        available_isl_laser_heads=0,
        isl_head_occupancy=None,
        tp_communication_time_s=None,
        ep_communication_time_s=None,
        pp_communication_time_s=None,
        kv_handoff_time_s=None,
        communication_bottleneck=None,
        placement_warnings=(),
        bottlenecks={},
        warnings=(message,),
        provenance={},
    )


def _model_preset(name: str) -> dict[str, Any]:
    presets: dict[str, dict[str, Any]] = {
        "llama2-7b": {
            "name": "llama2-7b",
            "family": "llama2",
            "parameter_count_b": 7.0,
            "layer_count": 32,
            "hidden_size": 4096,
            "attention_head_count": 32,
            "kv_head_count": 32,
            "intermediate_size": 11008,
            "max_context_tokens": 4096,
            "architecture_source": "preset_llama2",
        },
        "llama2-13b": {
            "name": "llama2-13b",
            "family": "llama2",
            "parameter_count_b": 13.0,
            "layer_count": 40,
            "hidden_size": 5120,
            "attention_head_count": 40,
            "kv_head_count": 40,
            "intermediate_size": 13824,
            "max_context_tokens": 4096,
            "architecture_source": "preset_llama2",
        },
        "llama2-70b": {
            "name": "llama2-70b",
            "family": "llama2",
            "parameter_count_b": 70.0,
            "layer_count": 80,
            "hidden_size": 8192,
            "attention_head_count": 64,
            "kv_head_count": 8,
            "intermediate_size": 28672,
            "max_context_tokens": 4096,
            "architecture_source": "preset_llama2",
        },
        "llama3-8b": {
            "name": "llama3-8b",
            "family": "llama3",
            "parameter_count_b": 8.0,
            "layer_count": 32,
            "hidden_size": 4096,
            "attention_head_count": 32,
            "kv_head_count": 8,
            "intermediate_size": 14336,
            "max_context_tokens": 8192,
            "architecture_source": "preset_llama3",
        },
        "llama3-70b": {
            "name": "llama3-70b",
            "family": "llama3",
            "parameter_count_b": 70.0,
            "layer_count": 80,
            "hidden_size": 8192,
            "attention_head_count": 64,
            "kv_head_count": 8,
            "intermediate_size": 28672,
            "max_context_tokens": 8192,
            "architecture_source": "preset_llama3",
        },
        "qwen2.5-7b": {
            "name": "qwen2.5-7b",
            "family": "qwen",
            "parameter_count_b": 7.6,
            "layer_count": 28,
            "hidden_size": 3584,
            "attention_head_count": 28,
            "kv_head_count": 4,
            "intermediate_size": 18944,
            "max_context_tokens": 32768,
            "architecture_source": "preset_qwen_review_required",
        },
        "qwen2.5-72b": {
            "name": "qwen2.5-72b",
            "family": "qwen",
            "parameter_count_b": 72.7,
            "layer_count": 80,
            "hidden_size": 8192,
            "attention_head_count": 64,
            "kv_head_count": 8,
            "intermediate_size": 29568,
            "max_context_tokens": 32768,
            "architecture_source": "preset_qwen_review_required",
        },
        "generic-305b-dense": {
            "name": "generic-305b-dense",
            "family": "generic",
            "parameter_count_b": 305.0,
            "layer_count": 96,
            "hidden_size": 12288,
            "attention_head_count": 96,
            "kv_head_count": 12,
            "intermediate_size": 49152,
            "max_context_tokens": 32768,
            "architecture_source": "generic_305b_dense_placeholder",
        },
        "generic-305b-moe-a32b": {
            "name": "generic-305b-moe-a32b",
            "family": "generic-moe",
            "parameter_count_b": 305.0,
            "active_parameter_count_b": 32.0,
            "layer_count": 80,
            "hidden_size": 8192,
            "attention_head_count": 64,
            "kv_head_count": 8,
            "expert_count": 128,
            "active_expert_count": 8,
            "intermediate_size": 28672,
            "max_context_tokens": 32768,
            "architecture_source": "generic_305b_moe_placeholder",
        },
        "deepseek-moe-671b": {
            "name": "deepseek-moe-671b",
            "family": "deepseek",
            "parameter_count_b": 671.0,
            "active_parameter_count_b": 37.0,
            "layer_count": 61,
            "hidden_size": 7168,
            "attention_head_count": 128,
            "kv_head_count": 128,
            "expert_count": 256,
            "active_expert_count": 8,
            "max_context_tokens": 128000,
            "architecture_source": "preset_deepseek_moe_review_required",
        },
        "trillion-dense-generic": {
            "name": "trillion-dense-generic",
            "family": "generic",
            "parameter_count_b": 1000.0,
            "layer_count": 128,
            "hidden_size": 18432,
            "attention_head_count": 144,
            "kv_head_count": 16,
            "intermediate_size": 73728,
            "max_context_tokens": 32768,
            "architecture_source": "generic_trillion_scale_placeholder",
        },
    }
    key = name.lower().replace("_", "-")
    if key not in presets:
        known = ", ".join(sorted(presets))
        raise ValueError(f"unknown model preset '{name}'. Known presets: {known}")
    return presets[key]


def _heuristic_architecture(parameter_count_b: float, family: str) -> dict[str, int]:
    if parameter_count_b <= 8:
        layers, hidden, heads = 32, 4096, 32
    elif parameter_count_b <= 16:
        layers, hidden, heads = 40, 5120, 40
    elif parameter_count_b <= 40:
        layers, hidden, heads = 48, 8192, 64
    elif parameter_count_b <= 90:
        layers, hidden, heads = 80, 8192, 64
    elif parameter_count_b <= 350:
        layers, hidden, heads = 96, 12288, 96
    elif parameter_count_b <= 800:
        layers, hidden, heads = 112, 16384, 128
    else:
        layers, hidden, heads = 128, 18432, 144
    kv_heads = heads
    if family.lower() in ("llama3", "qwen", "deepseek", "generic"):
        kv_heads = max(1, heads // 8)
    return {
        "layer_count": layers,
        "hidden_size": hidden,
        "attention_head_count": heads,
        "kv_head_count": kv_heads,
        "intermediate_size": hidden * 4,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
