"""Static-batch analytical Serving model for one logical L2 compute node."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..architecture import SatelliteHardwareProfile
from ..common import PerformanceMetrics, to_jsonable
from .model_profiles import ModelServingProfile, model_serving_profile


DEFAULT_PROMPT_LENGTH_GRID = (
    1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288, 1048576,
)
DEFAULT_CONTEXT_LENGTH_GRID = DEFAULT_PROMPT_LENGTH_GRID


@dataclass(frozen=True)
class LLMModelProfile:
    """A name-only reference to one frozen Serving model profile."""

    name: str = "llama3.3-70b"

    def __post_init__(self) -> None:
        resolved = model_serving_profile(self.name)
        object.__setattr__(self, "name", resolved.name)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str | None) -> "LLMModelProfile":
        if data is None:
            return cls()
        if isinstance(data, str):
            return cls(data)
        unknown = sorted(set(data) - {"name", "preset"})
        if unknown:
            raise ValueError("serving.model accepts only name/preset; unsupported fields: " + ", ".join(unknown))
        return cls(str(data.get("name") or data.get("preset") or "llama3.3-70b"))

    @property
    def profile(self) -> ModelServingProfile:
        return model_serving_profile(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class ServingParallelism:
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    expert_parallel_degree: int = 1
    tensor_parallel_collectives_per_layer: int = 2
    expert_parallel_imbalance_factor: float = 1.0
    expert_parallel_startup_us: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "tensor_parallel_degree", "pipeline_parallel_degree",
            "expert_parallel_degree",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"serving.parallelism.{name} must be a positive integer")
        if type(self.tensor_parallel_collectives_per_layer) is not int or self.tensor_parallel_collectives_per_layer < 0:
            raise ValueError("tensor_parallel_collectives_per_layer must be a non-negative integer")
        for name, minimum in (('expert_parallel_imbalance_factor', 1.), ('expert_parallel_startup_us', 0.)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
                raise ValueError(f'serving.parallelism.{name} must be finite and >= {minimum}')

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ServingParallelism":
        data = data or {}
        aliases = {"tp": "tensor_parallel_degree", "pp": "pipeline_parallel_degree", "ep": "expert_parallel_degree"}
        normalized = {aliases.get(key, key): value for key, value in data.items()}
        unknown = sorted(set(normalized) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unsupported serving.parallelism fields: " + ", ".join(unknown))
        return cls(**normalized)

    @property
    def stage_parallel_node_count(self) -> int:
        return math.lcm(self.tensor_parallel_degree, self.expert_parallel_degree)

    @property
    def parallel_node_count(self) -> int:
        return self.pipeline_parallel_degree * self.stage_parallel_node_count

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class L2FormationSpec:
    member_count: int = 1
    topology: str = "ring"
    link_bandwidth_gbps: float = 0.0
    link_propagation_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if type(self.member_count) is not int or self.member_count < 1:
            raise ValueError("serving.formation.member_count must be a positive integer")
        if self.topology not in {"ring", "switch", "fully_connected"}:
            raise ValueError("serving.formation.topology must be ring, switch, or fully_connected")
        if not math.isfinite(self.link_bandwidth_gbps) or self.link_bandwidth_gbps < 0.0:
            raise ValueError("serving.formation.link_bandwidth_gbps must be finite and non-negative")
        if not math.isfinite(self.link_propagation_delay_s) or self.link_propagation_delay_s < 0.0:
            raise ValueError("serving.formation.link_propagation_delay_s must be finite and non-negative")
        if self.member_count > 1 and self.link_bandwidth_gbps <= 0.0:
            raise ValueError("serving.formation.link_bandwidth_gbps must be positive when member_count > 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "L2FormationSpec":
        data = data or {}
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unsupported serving.formation fields: " + ", ".join(unknown))
        return cls(
            member_count=data.get("member_count", 1),
            topology=str(data.get("topology", "ring")).lower(),
            link_bandwidth_gbps=float(data.get("link_bandwidth_gbps", 0.0)),
            link_propagation_delay_s=float(data.get("link_propagation_delay_s", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ServingSpec:
    model: LLMModelProfile = field(default_factory=LLMModelProfile)
    parallelism: ServingParallelism = field(default_factory=ServingParallelism)
    formation: L2FormationSpec = field(default_factory=L2FormationSpec)
    prefill_compute_efficiency: float = 0.5
    decode_memory_efficiency: float = 0.7
    prompt_length_grid: tuple[int, ...] = DEFAULT_PROMPT_LENGTH_GRID
    context_length_grid: tuple[int, ...] = DEFAULT_CONTEXT_LENGTH_GRID
    max_concurrency: int = 1
    hardware: SatelliteHardwareProfile | None = None
    external_profile: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.external_profile is not None and not isinstance(self.external_profile,dict):
            raise ValueError('serving.external_profile must be a measurement mapping')
        for name in ("prefill_compute_efficiency", "decode_memory_efficiency"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"serving.{name} must be in (0, 1]")
        _validate_length_grid("prompt_length_grid", self.prompt_length_grid)
        _validate_length_grid("context_length_grid", self.context_length_grid)
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("serving.max_concurrency must be a positive integer")
        _validate_parallelism(self.model.profile, self.parallelism, self.formation)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ServingSpec":
        data = data or {}
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unsupported ServingSpec fields: " + ", ".join(unknown))
        hardware = data.get("hardware")
        return cls(
            model=LLMModelProfile.from_dict(data.get("model")),
            parallelism=ServingParallelism.from_dict(data.get("parallelism")),
            formation=L2FormationSpec.from_dict(data.get("formation")),
            prefill_compute_efficiency=float(data.get("prefill_compute_efficiency", 0.5)),
            decode_memory_efficiency=float(data.get("decode_memory_efficiency", 0.7)),
            prompt_length_grid=tuple(data.get("prompt_length_grid", DEFAULT_PROMPT_LENGTH_GRID)),
            context_length_grid=tuple(data.get("context_length_grid", DEFAULT_CONTEXT_LENGTH_GRID)),
            max_concurrency=data.get("max_concurrency", 1),
            hardware=SatelliteHardwareProfile.from_dict(hardware) if isinstance(hardware, dict) else hardware,
            external_profile=data.get('external_profile'),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "parallelism": self.parallelism.to_dict(),
            "formation": self.formation.to_dict(),
            "prefill_compute_efficiency": self.prefill_compute_efficiency,
            "decode_memory_efficiency": self.decode_memory_efficiency,
            "prompt_length_grid": self.prompt_length_grid,
            "context_length_grid": self.context_length_grid,
            "max_concurrency": self.max_concurrency,
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "external_profile": self.external_profile,
        }


@dataclass(frozen=True)
class ServingPerformanceProfile:
    evaluated: bool
    feasible: bool
    model_name: str
    model: dict[str, Any]
    hardware: dict[str, Any]
    parallelism: dict[str, Any]
    formation: dict[str, Any]
    prefill_rate_table: tuple[dict[str, Any], ...]
    decode_rate_table: tuple[dict[str, Any], ...]
    max_concurrency: int
    kv_cache_bytes_per_token: float
    kv_cache_capacity_tokens: int | None
    max_context_tokens: int
    physical_satellite_count: int
    internal_communication_accounting: str
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_performance_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            evaluated=self.evaluated,
            model_weight_memory_gb=self.model.get("total_weight_bytes", 0.0) / 1e9,
            kv_cache_capacity_tokens=self.kv_cache_capacity_tokens,
            satellite_count_used=self.physical_satellite_count,
            feasible=self.feasible,
            bottleneck="prefill:compute, decode:memory",
            details=self.to_dict(),
            warnings=self.warnings,
        )


ServingThroughputEstimate = ServingPerformanceProfile


def estimate_serving_throughput(
    serving: ServingSpec,
    *,
    hardware: SatelliteHardwareProfile | None = None,
) -> ServingPerformanceProfile:
    """Generate homogeneous static-batch rate tables for one logical node."""

    resolved_hardware = hardware or serving.hardware
    if resolved_hardware is None:
        return _unevaluated(serving, "serving.hardware is not configured")
    model = serving.model.profile
    prompt_lengths = _resolved_lengths(serving.prompt_length_grid, model.max_context_tokens)
    context_lengths = _resolved_lengths(serving.context_length_grid, model.max_context_tokens)
    stage_nodes = serving.parallelism.stage_parallel_node_count
    effective_flops = (
        stage_nodes
        * resolved_hardware.total_peak_compute_tflops
        * 1e12
        * serving.prefill_compute_efficiency
    )
    effective_memory_bytes_per_s = (
        stage_nodes
        * resolved_hardware.total_fast_mem_bw_gbs
        * 1e9
        * serving.decode_memory_efficiency
    )
    if effective_flops <= 0.0 or effective_memory_bytes_per_s <= 0.0:
        return _unevaluated(serving, "hardware compute and memory bandwidth must be positive")
    prefill_table = tuple(
        {
            "prompt_length": length,
            "concurrency": concurrency,
            "prefill_tokens_per_s": length / (
                concurrency * prefill_flops(model.name, length) / effective_flops
                + _formation_communication_time_s(
                    serving, model, concurrency * length
                )
            ),
        }
        for length in prompt_lengths
        for concurrency in range(1, serving.max_concurrency + 1)
    )
    decode_table = tuple(
        {
            "context_length": length,
            "concurrency": concurrency,
            "decode_tokens_per_s": 1.0 / (
                (
                    model.active_weight_bytes
                    + concurrency * (length + 1) * model.kv_cache_bytes_per_token
                )
                / effective_memory_bytes_per_s
                + _formation_communication_time_s(serving, model, concurrency)
            ),
        }
        for length in context_lengths
        for concurrency in range(1, serving.max_concurrency + 1)
    )
    capacity = math.floor(
        serving.formation.member_count
        * resolved_hardware.storage_capacity_tb
        * 1e12
        / model.kv_cache_bytes_per_token
    )
    per_member_weight_bytes = model.total_weight_bytes / serving.formation.member_count
    member_fast_memory_bytes = resolved_hardware.total_fast_mem_gb * 1e9
    feasible = per_member_weight_bytes <= member_fast_memory_bytes
    warnings = () if feasible else ("model weights do not fit formation member fast memory",)
    _validate_rate_tables(
        prefill_table,
        decode_table,
        maximum=model.max_context_tokens,
        max_concurrency=serving.max_concurrency,
    )
    profile = ServingPerformanceProfile(
        evaluated=True,
        feasible=feasible,
        model_name=model.name,
        model={
            "name": model.name,
            "family": model.family,
            "total_parameter_count": model.total_parameter_count,
            "active_parameter_count": model.active_parameter_count,
            "total_weight_bytes": model.total_weight_bytes,
            "prefill_linear_flops_per_token": model.prefill_linear_flops_per_token,
            "prefill_quadratic_flops_coefficient": model.prefill_quadratic_flops_coefficient,
            "architecture_source": model.architecture_source,
        },
        hardware=resolved_hardware.to_dict(),
        parallelism=serving.parallelism.to_dict(),
        formation=serving.formation.to_dict(),
        prefill_rate_table=prefill_table,
        decode_rate_table=decode_table,
        max_concurrency=serving.max_concurrency,
        kv_cache_bytes_per_token=model.kv_cache_bytes_per_token,
        kv_cache_capacity_tokens=capacity,
        max_context_tokens=model.max_context_tokens,
        physical_satellite_count=serving.formation.member_count,
        internal_communication_accounting="embedded",
        warnings=warnings,
        provenance={
            "execution_model": "homogeneous_static_batch",
            "execution_assumptions": ("prefill_decode_colocated",),
            "rate_semantics": "per_request",
            "concurrency_range": (1, serving.max_concurrency),
            "prefill_model": "compute_dominant_A_times_P_plus_B_times_P_squared",
            "decode_model": "memory_dominant_active_weights_plus_kv",
            "table_entries": "length_concurrency_and_per_request_tokens_per_s",
            "formation_communication": "topology_aware_alpha_beta_embedded",
            "model_approximation_notes": model.approximation_notes,
        },
    )
    if serving.external_profile is not None:
        from .external import from_astermind_measurements
        return from_astermind_measurements(profile, serving.external_profile)
    return profile


def prefill_flops(model_name: str, prompt_length: int) -> float:
    model = model_serving_profile(model_name)
    if prompt_length < 0:
        raise ValueError("prompt_length must be non-negative")
    return (
        model.prefill_linear_flops_per_token * prompt_length
        + model.prefill_quadratic_flops_coefficient * prompt_length**2
    )


def _validate_parallelism(
    model: ModelServingProfile,
    parallelism: ServingParallelism,
    formation: L2FormationSpec,
) -> None:
    if parallelism.parallel_node_count != formation.member_count:
        raise ValueError(
            "serving parallel_node_count must equal formation.member_count: "
            f"{parallelism.parallel_node_count} != {formation.member_count}"
        )
    if not model.is_moe and parallelism.expert_parallel_degree != 1:
        raise ValueError("expert_parallel_degree must be 1 for non-MoE models")
    if model.is_moe:
        ep = parallelism.expert_parallel_degree
        if ep > formation.member_count or formation.member_count % ep != 0:
            raise ValueError("formation.member_count must be divisible by expert_parallel_degree")
        if model.expert_count % ep != 0:
            raise ValueError("model expert_count must be divisible by expert_parallel_degree")


def _validate_length_grid(name: str, grid: tuple[int, ...]) -> None:
    if not grid or any(type(value) is not int or value < 1 for value in grid):
        raise ValueError(f"serving.{name} must contain positive integers")
    if tuple(sorted(set(grid))) != grid:
        raise ValueError(f"serving.{name} must be strictly increasing")


def _resolved_lengths(grid: tuple[int, ...], maximum: int) -> tuple[int, ...]:
    return tuple(sorted({*(value for value in grid if value <= maximum), maximum}))


def _formation_communication_time_s(
    serving: ServingSpec,
    model: ModelServingProfile,
    token_count: int,
) -> float:
    formation = serving.formation
    parallelism = serving.parallelism
    if formation.member_count <= 1:
        return 0.0
    message_bytes = token_count * model.hidden_size * model.activation_bytes
    communication_time = 0.0
    if parallelism.tensor_parallel_degree > 1:
        communication_time += (
            model.layer_count
            * parallelism.tensor_parallel_collectives_per_layer
            * _collective_time_s(
                formation,
                message_bytes,
                degree=parallelism.tensor_parallel_degree,
            )
        )
    if parallelism.expert_parallel_degree > 1:
        expert_time = (
            2
            * model.layer_count
            * _collective_time_s(
                formation,
                message_bytes,
                degree=parallelism.expert_parallel_degree,
            )
        )
        communication_time += expert_time * parallelism.expert_parallel_imbalance_factor
        communication_time += model.layer_count * parallelism.expert_parallel_startup_us * 1e-6
    pp_transfers = max(0, parallelism.pipeline_parallel_degree - 1)
    pp_time = pp_transfers * _point_to_point_time_s(formation, message_bytes)
    return communication_time + pp_time


def _collective_time_s(
    formation: L2FormationSpec,
    message_bytes: float,
    *,
    degree: int,
) -> float:
    bandwidth_bytes_per_s = formation.link_bandwidth_gbps * 1e9 / 8.0
    volume = 2.0 * (degree - 1) / degree * message_bytes
    if formation.topology == "ring":
        rounds = 2 * (degree - 1)
    elif formation.topology == "switch":
        rounds = 2
    else:
        rounds = max(1, math.ceil(math.log2(degree)))
    return rounds * formation.link_propagation_delay_s + volume / bandwidth_bytes_per_s


def _point_to_point_time_s(formation: L2FormationSpec, message_bytes: float) -> float:
    bandwidth_bytes_per_s = formation.link_bandwidth_gbps * 1e9 / 8.0
    hops = max(1, formation.member_count // 2) if formation.topology == "ring" else 1
    return hops * formation.link_propagation_delay_s + message_bytes / bandwidth_bytes_per_s


def _validate_rate_tables(
    prefill: tuple[dict[str, Any], ...],
    decode: tuple[dict[str, Any], ...],
    *,
    maximum: int,
    max_concurrency: int,
) -> None:
    expected_concurrency = tuple(range(1, max_concurrency + 1))
    for rows, length_key, rate_key in (
        (prefill, "prompt_length", "prefill_tokens_per_s"),
        (decode, "context_length", "decode_tokens_per_s"),
    ):
        if any(
            not math.isfinite(float(row[rate_key])) or float(row[rate_key]) <= 0.0
            for row in rows
        ):
            raise ValueError("Serving rate table rates must be finite and positive")
        lengths = tuple(sorted({int(row[length_key]) for row in rows}))
        if not lengths or lengths[-1] != maximum:
            raise ValueError("Serving rate table must cover max context")
        for length in lengths:
            concurrency = tuple(
                int(row["concurrency"])
                for row in rows
                if int(row[length_key]) == length
            )
            if concurrency != expected_concurrency:
                raise ValueError("Serving rate table concurrency slices must be complete and sorted")
        for concurrency in expected_concurrency:
            sliced = tuple(row for row in rows if int(row["concurrency"]) == concurrency)
            sliced_lengths = tuple(int(row[length_key]) for row in sliced)
            if sliced_lengths != lengths:
                raise ValueError("Serving rate table lengths must be unique and sorted")
            times = tuple(
                (
                    row[length_key] / row[rate_key]
                    if rate_key == "prefill_tokens_per_s"
                    else 1.0 / row[rate_key]
                )
                for row in sliced
            )
            if any(right < left for left, right in zip(times, times[1:])):
                raise ValueError("Serving rate table service time must be non-decreasing")


def _unevaluated(serving: ServingSpec, message: str) -> ServingPerformanceProfile:
    model = serving.model.profile
    return ServingPerformanceProfile(
        evaluated=False,
        feasible=False,
        model_name=model.name,
        model={"name": model.name},
        hardware={},
        parallelism=serving.parallelism.to_dict(),
        formation=serving.formation.to_dict(),
        prefill_rate_table=(),
        decode_rate_table=(),
        max_concurrency=serving.max_concurrency,
        kv_cache_bytes_per_token=model.kv_cache_bytes_per_token,
        kv_cache_capacity_tokens=None,
        max_context_tokens=model.max_context_tokens,
        physical_satellite_count=serving.formation.member_count,
        internal_communication_accounting="embedded",
        warnings=(message,),
        provenance={},
    )


__all__ = [
    "DEFAULT_CONTEXT_LENGTH_GRID",
    "DEFAULT_PROMPT_LENGTH_GRID",
    "LLMModelProfile",
    "L2FormationSpec",
    "ServingParallelism",
    "ServingPerformanceProfile",
    "ServingSpec",
    "ServingThroughputEstimate",
    "estimate_serving_throughput",
    "prefill_flops",
]
