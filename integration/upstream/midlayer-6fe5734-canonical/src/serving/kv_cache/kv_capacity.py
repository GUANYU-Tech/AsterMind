"""Analytical HBM-to-KV capacity contract for production Serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Optional

from src.kernel.hardware_spec import HWSpec
from src.serving.kv_cache.model_residency import model_rank_residencies
from src.serving.kv_cache.vllm_allocation import (
    VLLMKVAllocation,
    allocate_vllm_kv_cache,
)
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.workload.model_config import ModelConfig


_PROVENANCE_FIELDS = {"source_kind", "source_reference", "source_sha256"}
_POLICY_FIELDS = {
    "pool_name", "binding", "gpu_memory_utilization", "non_kv_reserves",
    "reserve_scope", "completeness_assertion",
}
_BINDING_FIELDS = {
    "hardware_id", "hardware_config_sha256", "model_config_sha256",
    "deployment", "worker_role", "tensor_parallel_size",
    "pipeline_parallel_size", "expert_parallel_size", "max_model_len",
    "max_batched_tokens", "max_requests", "kv_block_size_tokens",
}
_UTILIZATION_FIELDS = {"numerator", "denominator", "provenance"}
_RESERVE_FIELDS = {"kind", "bytes", "provenance"}
_REQUIRED_RESERVE_KINDS = {
    "activation", "cuda_graph", "communication", "allocator",
    "fragmentation", "framework",
}
_SOURCE_KINDS = {
    "runtime_configuration", "architecture_document", "compiler_source",
    "framework_source", "microbenchmark",
}


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    normalized = {}
    for key, value in pairs:
        if key in normalized:
            raise ValueError(
                f"duplicate runtime memory policy JSON field: {key!r}")
        normalized[key] = value
    return normalized


def _require_integer(name: str, value: object, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"KV capacity {name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"KV capacity {name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"KV capacity {name} cannot be negative")
    return value


@dataclass(frozen=True)
class CapacityEvidence:
    """Provenance for one runtime memory-policy value."""

    source_kind: str
    source_reference: str
    source_sha256: Optional[str]

    def __post_init__(self) -> None:
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(
                f"unsupported capacity evidence source_kind: "
                f"{self.source_kind!r}")
        if (not isinstance(self.source_reference, str)
                or not self.source_reference):
            raise ValueError(
                "capacity evidence source_reference must be non-empty")
        if self.source_sha256 is not None and (
                not isinstance(self.source_sha256, str)
                or len(self.source_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.source_sha256)):
            raise ValueError(
                "capacity evidence source_sha256 must be null or a lowercase "
                "SHA256")


@dataclass(frozen=True)
class RuntimeMemoryReserve:
    """One explicit non-KV runtime allocation in bytes per physical rank."""

    kind: str
    bytes: int
    provenance: CapacityEvidence

    def __post_init__(self) -> None:
        if self.kind not in _REQUIRED_RESERVE_KINDS:
            raise ValueError(
                f"unsupported runtime reserve kind: {self.kind!r}")
        _require_integer(
            f"{self.kind} reserve bytes", self.bytes, positive=False)
        if not isinstance(self.provenance, CapacityEvidence):
            raise TypeError("runtime reserve provenance must be CapacityEvidence")


@dataclass(frozen=True)
class RuntimeMemoryPolicyBinding:
    """Exact runtime identity for which a reserve policy is valid."""

    hardware_id: str
    hardware_config_sha256: str
    model_config_sha256: str
    deployment: str
    worker_role: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    max_model_len: int
    max_batched_tokens: int
    max_requests: int
    kv_block_size_tokens: int

    def __post_init__(self) -> None:
        for name in ("hardware_id", "deployment", "worker_role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"runtime memory policy binding {name} must be non-empty")
        for name in ("hardware_config_sha256", "model_config_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef"
                           for character in value)):
                raise ValueError(
                    f"runtime memory policy binding {name} must be a "
                    "lowercase SHA256")
        for name in (
                "tensor_parallel_size", "pipeline_parallel_size",
                "expert_parallel_size", "max_model_len",
                "max_batched_tokens", "max_requests",
                "kv_block_size_tokens"):
            _require_integer(name, getattr(self, name), positive=True)


@dataclass(frozen=True)
class RuntimeMemoryPolicy:
    """Complete per-rank runtime memory policy for one scheduler pool."""

    pool_name: str
    binding: RuntimeMemoryPolicyBinding
    gpu_memory_utilization_numerator: int
    gpu_memory_utilization_denominator: int
    gpu_memory_utilization_provenance: CapacityEvidence
    non_kv_reserves: tuple[RuntimeMemoryReserve, ...]
    reserve_scope: str
    completeness_assertion: str

    def __post_init__(self) -> None:
        if not isinstance(self.pool_name, str) or not self.pool_name:
            raise ValueError("runtime memory policy pool_name must be non-empty")
        if not isinstance(self.binding, RuntimeMemoryPolicyBinding):
            raise TypeError(
                "runtime memory policy binding must be RuntimeMemoryPolicyBinding")
        numerator = _require_integer(
            "gpu_memory_utilization numerator",
            self.gpu_memory_utilization_numerator,
            positive=True,
        )
        denominator = _require_integer(
            "gpu_memory_utilization denominator",
            self.gpu_memory_utilization_denominator,
            positive=True,
        )
        if numerator > denominator:
            raise ValueError("gpu_memory_utilization cannot exceed one")
        if not isinstance(
                self.gpu_memory_utilization_provenance, CapacityEvidence):
            raise TypeError(
                "gpu_memory_utilization provenance must be CapacityEvidence")
        if not isinstance(self.non_kv_reserves, tuple):
            raise TypeError("non_kv_reserves must be a tuple")
        if any(not isinstance(item, RuntimeMemoryReserve)
               for item in self.non_kv_reserves):
            raise TypeError(
                "non_kv_reserves must contain RuntimeMemoryReserve values")
        kinds = tuple(item.kind for item in self.non_kv_reserves)
        if set(kinds) != _REQUIRED_RESERVE_KINDS or len(kinds) != len(set(kinds)):
            raise ValueError(
                "runtime policy must account for each non-KV reserve kind "
                f"exactly once: {sorted(_REQUIRED_RESERVE_KINDS)}")
        if self.reserve_scope != "verified_uniform_across_pool_ranks":
            raise ValueError(
                "runtime policy reserve_scope must be "
                "'verified_uniform_across_pool_ranks'")
        if self.completeness_assertion != "all_non_kv_reserves_for_bound_runtime":
            raise ValueError(
                "runtime policy completeness_assertion must be "
                "'all_non_kv_reserves_for_bound_runtime'")

    @property
    def non_kv_reserve_bytes(self) -> int:
        return sum(reserve.bytes for reserve in self.non_kv_reserves)

    def utilized_hbm_bytes(self, physical_hbm_bytes: int) -> int:
        _require_integer("physical_hbm_bytes", physical_hbm_bytes, positive=True)
        return (
            physical_hbm_bytes * self.gpu_memory_utilization_numerator
            // self.gpu_memory_utilization_denominator
        )


@dataclass(frozen=True)
class RuntimeMemoryPolicyConfig:
    """Strict versioned runtime-memory policy file."""

    source_path: str
    source_sha256: str
    policies: Mapping[str, RuntimeMemoryPolicy]

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("runtime memory policy source_path must be non-empty")
        if (not isinstance(self.source_sha256, str)
                or len(self.source_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.source_sha256)):
            raise ValueError(
                "runtime memory policy source_sha256 must be a lowercase SHA256")
        if not isinstance(self.policies, Mapping) or not self.policies:
            raise ValueError("runtime memory policy requires scheduler pools")
        normalized = dict(self.policies)
        for pool_name, policy in normalized.items():
            if not isinstance(pool_name, str) or not pool_name:
                raise ValueError(
                    "runtime memory policy keys must be non-empty strings")
            if not isinstance(policy, RuntimeMemoryPolicy):
                raise TypeError(
                    "runtime memory policy values must be RuntimeMemoryPolicy")
            if policy.pool_name != pool_name:
                raise ValueError(
                    f"runtime policy key {pool_name!r} disagrees with "
                    f"pool_name {policy.pool_name!r}")
        object.__setattr__(self, "policies", MappingProxyType(normalized))

    def policy_for_pool(self, pool_name: str) -> RuntimeMemoryPolicy:
        if pool_name not in self.policies:
            raise ValueError(
                f"runtime memory policy has no scheduler pool {pool_name!r}")
        return self.policies[pool_name]

    def validate_scheduler_pools(self, pool_names: Iterable[str]) -> None:
        normalized = tuple(pool_names)
        if (not normalized
                or any(not isinstance(name, str) or not name
                       for name in normalized)):
            raise ValueError(
                "runtime memory scheduler pools must be non-empty strings")
        if len(normalized) != len(set(normalized)):
            raise ValueError("runtime memory scheduler pool names must be unique")
        expected = set(self.policies)
        actual = set(normalized)
        if actual != expected:
            raise ValueError(
                "runtime memory policy must cover scheduler pools exactly: "
                f"missing={sorted(actual - expected)}, "
                f"unexpected={sorted(expected - actual)}")


@dataclass(frozen=True)
class KVCapacityRankBudget:
    """Analytical byte accounting for one physical model rank."""

    rank: int
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
    expert_parallel_rank: int
    layer_start: int
    layer_end: int
    physical_hbm_bytes: int
    utilized_hbm_bytes: int
    model_weight_resident_bytes: int
    non_kv_reserve_bytes: int
    available_kv_bytes: int

    def __post_init__(self) -> None:
        for name in (
                "rank", "tensor_parallel_rank", "pipeline_parallel_rank",
                "expert_parallel_rank", "layer_start", "layer_end",
                "physical_hbm_bytes", "utilized_hbm_bytes",
                "model_weight_resident_bytes", "non_kv_reserve_bytes",
                "available_kv_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"rank budget {name} must be an integer")
            if value < 0:
                raise ValueError(f"rank budget {name} cannot be negative")
        if self.available_kv_bytes != (
                self.utilized_hbm_bytes
                - self.model_weight_resident_bytes
                - self.non_kv_reserve_bytes):
            raise ValueError("rank available KV bytes disagree with components")
        if self.layer_end <= self.layer_start:
            raise ValueError("rank budget layer range must be non-empty")
        if self.utilized_hbm_bytes > self.physical_hbm_bytes:
            raise ValueError("rank utilized HBM bytes exceed physical HBM")
        if self.available_kv_bytes <= 0:
            raise ValueError("rank available KV bytes must be positive")


@dataclass(frozen=True)
class KVCapacityPlan:
    """Resolved capacity consumed by one native framework scheduler."""

    pool_name: str
    source_mode: str
    kv_block_size_tokens: int
    kv_bytes_per_token: int
    kv_block_bytes: int
    num_kv_blocks: int
    hardware_id: Optional[str]
    model_config_sha256: Optional[str]
    tensor_parallel_size: Optional[int]
    pipeline_parallel_size: Optional[int]
    expert_parallel_size: Optional[int]
    worker_role: Optional[str]
    rank_budgets: tuple[KVCapacityRankBudget, ...]
    native_allocation: Optional[VLLMKVAllocation]

    def __post_init__(self) -> None:
        if not isinstance(self.pool_name, str) or not self.pool_name:
            raise ValueError("KV capacity plan pool_name must be non-empty")
        if self.source_mode not in (
                "analytical", "debug_explicit_block_count"):
            raise ValueError(
                f"unknown KV capacity source_mode: {self.source_mode!r}")
        for name in (
                "kv_block_size_tokens", "kv_bytes_per_token",
                "kv_block_bytes", "num_kv_blocks"):
            _require_integer(name, getattr(self, name), positive=True)
        if self.kv_block_bytes != (
                self.kv_block_size_tokens * self.kv_bytes_per_token):
            raise ValueError(
                "KV capacity block bytes disagree with block size and bytes "
                "per token")
        if self.source_mode == "analytical":
            if (not self.rank_budgets
                    or self.native_allocation is None
                    or self.native_allocation.num_kv_blocks
                    != self.num_kv_blocks):
                raise ValueError(
                    "analytical KV capacity requires rank budgets and the "
                    "matching native allocation")
            if any(value is None for value in (
                    self.hardware_id, self.model_config_sha256,
                    self.tensor_parallel_size, self.pipeline_parallel_size,
                    self.expert_parallel_size, self.worker_role)):
                raise ValueError(
                    "analytical KV capacity requires all bound identities")
            if (not isinstance(self.hardware_id, str) or not self.hardware_id
                    or not isinstance(self.worker_role, str)
                    or not self.worker_role):
                raise ValueError(
                    "analytical KV capacity identities must be non-empty")
            if (not isinstance(self.model_config_sha256, str)
                    or len(self.model_config_sha256) != 64
                    or any(character not in "0123456789abcdef"
                           for character in self.model_config_sha256)):
                raise ValueError(
                    "analytical KV capacity model identity must be a SHA256")
            for name in (
                    "tensor_parallel_size", "pipeline_parallel_size",
                    "expert_parallel_size"):
                _require_integer(name, getattr(self, name), positive=True)
            expected_rank_count = (
                self.tensor_parallel_size * self.pipeline_parallel_size)
            if len(self.rank_budgets) != expected_rank_count:
                raise ValueError(
                    "analytical KV capacity rank count disagrees with TP x PP")
            if tuple(budget.rank for budget in self.rank_budgets) != tuple(
                    range(expected_rank_count)):
                raise ValueError(
                    "analytical KV capacity ranks must be contiguous and "
                    "ordered")
            for budget in self.rank_budgets:
                expected_tp_rank = budget.rank % self.tensor_parallel_size
                expected_pp_rank = budget.rank // self.tensor_parallel_size
                expected_ep_rank = (
                    expected_tp_rank if self.expert_parallel_size > 1 else 0)
                if (budget.tensor_parallel_rank != expected_tp_rank
                        or budget.pipeline_parallel_rank != expected_pp_rank
                        or budget.expert_parallel_rank != expected_ep_rank):
                    raise ValueError(
                        "analytical KV capacity rank coordinates disagree "
                        "with TP/PP/EP")
            if (self.expert_parallel_size not in (
                    1, self.tensor_parallel_size)):
                raise ValueError(
                    "analytical KV capacity supports EP disabled or EP=TP")
            if (len(self.native_allocation.worker_num_kv_blocks)
                    != expected_rank_count):
                raise ValueError(
                    "native KV allocation worker count disagrees with TP x PP")
        elif self.source_mode == "debug_explicit_block_count":
            if self.rank_budgets or self.native_allocation is not None:
                raise ValueError(
                    "debug explicit capacity cannot claim analytical evidence")
            if any(value is not None for value in (
                    self.hardware_id, self.model_config_sha256,
                    self.tensor_parallel_size, self.pipeline_parallel_size,
                    self.expert_parallel_size, self.worker_role)):
                raise ValueError(
                    "debug explicit capacity cannot claim analytical bindings")
        else:
            raise AssertionError("validated KV capacity mode disappeared")


def _load_provenance(raw: object, label: str) -> CapacityEvidence:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be a mapping")
    if set(raw) != _PROVENANCE_FIELDS:
        raise ValueError(
            f"{label} fields must be exactly {sorted(_PROVENANCE_FIELDS)}")
    return CapacityEvidence(
        source_kind=raw["source_kind"],
        source_reference=raw["source_reference"],
        source_sha256=raw["source_sha256"],
    )


def load_runtime_memory_policy(
    path: str | Path,
) -> RuntimeMemoryPolicyConfig:
    """Load a complete, provenance-bearing runtime memory policy."""

    source_path = Path(path)
    payload = source_path.read_bytes()
    raw = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(raw, dict):
        raise TypeError("runtime memory policy root must be a mapping")
    expected_root_fields = {"schema_version", "scheduler_pools"}
    if set(raw) != expected_root_fields:
        raise ValueError(
            "runtime memory policy fields must be exactly "
            f"{sorted(expected_root_fields)}, got {sorted(raw)}")
    _require_integer("schema_version", raw["schema_version"], positive=True)
    if raw["schema_version"] != 2:
        raise ValueError("runtime memory policy schema_version must be 2")
    entries = raw["scheduler_pools"]
    if not isinstance(entries, list) or not entries:
        raise TypeError(
            "runtime memory policy scheduler_pools must be a non-empty list")
    policies = {}
    for index, entry in enumerate(entries):
        label = f"runtime memory policy scheduler_pools[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{label} must be a mapping")
        if set(entry) != _POLICY_FIELDS:
            raise ValueError(
                f"{label} fields must be exactly {sorted(_POLICY_FIELDS)}, "
                f"got {sorted(entry)}")
        utilization = entry["gpu_memory_utilization"]
        if not isinstance(utilization, dict):
            raise TypeError(f"{label}.gpu_memory_utilization must be a mapping")
        if set(utilization) != _UTILIZATION_FIELDS:
            raise ValueError(
                f"{label}.gpu_memory_utilization fields must be exactly "
                f"{sorted(_UTILIZATION_FIELDS)}")
        binding_raw = entry["binding"]
        if not isinstance(binding_raw, dict):
            raise TypeError(f"{label}.binding must be a mapping")
        if set(binding_raw) != _BINDING_FIELDS:
            raise ValueError(
                f"{label}.binding fields must be exactly "
                f"{sorted(_BINDING_FIELDS)}")
        reserves_raw = entry["non_kv_reserves"]
        if not isinstance(reserves_raw, list):
            raise TypeError(f"{label}.non_kv_reserves must be a list")
        reserves = []
        for reserve_index, reserve_raw in enumerate(reserves_raw):
            reserve_label = f"{label}.non_kv_reserves[{reserve_index}]"
            if not isinstance(reserve_raw, dict):
                raise TypeError(f"{reserve_label} must be a mapping")
            if set(reserve_raw) != _RESERVE_FIELDS:
                raise ValueError(
                    f"{reserve_label} fields must be exactly "
                    f"{sorted(_RESERVE_FIELDS)}")
            reserves.append(RuntimeMemoryReserve(
                kind=reserve_raw["kind"],
                bytes=reserve_raw["bytes"],
                provenance=_load_provenance(
                    reserve_raw["provenance"],
                    f"{reserve_label}.provenance",
                ),
            ))
        policy = RuntimeMemoryPolicy(
            pool_name=entry["pool_name"],
            binding=RuntimeMemoryPolicyBinding(**binding_raw),
            gpu_memory_utilization_numerator=utilization["numerator"],
            gpu_memory_utilization_denominator=utilization["denominator"],
            gpu_memory_utilization_provenance=_load_provenance(
                utilization["provenance"],
                f"{label}.gpu_memory_utilization.provenance",
            ),
            non_kv_reserves=tuple(reserves),
            reserve_scope=entry["reserve_scope"],
            completeness_assertion=entry["completeness_assertion"],
        )
        if policy.pool_name in policies:
            raise ValueError(
                f"duplicate runtime memory policy pool_name: "
                f"{policy.pool_name!r}")
        policies[policy.pool_name] = policy
    return RuntimeMemoryPolicyConfig(
        source_path=str(source_path.resolve()),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        policies=policies,
    )


def resolve_analytical_kv_capacity_plan(
    pool_name: str,
    *,
    hardware: HWSpec,
    hardware_config_sha256: str,
    model: ModelConfig,
    parallelism: ParallelConfig,
    deployment: str,
    worker_role: str,
    kv_block_size_tokens: int,
    max_model_len: int,
    max_batched_tokens: int,
    max_requests: int,
    policy: RuntimeMemoryPolicy,
    include_mtp: bool = False,
    allocator: Callable[..., VLLMKVAllocation] = allocate_vllm_kv_cache,
) -> KVCapacityPlan:
    """Derive rank budgets and ask pinned vLLM for the final block count."""

    if not isinstance(pool_name, str) or not pool_name:
        raise ValueError("KV capacity pool_name must be non-empty")
    if not isinstance(hardware, HWSpec):
        raise TypeError("analytical KV capacity requires HWSpec")
    if not isinstance(model, ModelConfig):
        raise TypeError("analytical KV capacity requires ModelConfig")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("analytical KV capacity requires ParallelConfig")
    if not isinstance(policy, RuntimeMemoryPolicy):
        raise TypeError("analytical KV capacity requires RuntimeMemoryPolicy")
    if not isinstance(include_mtp, bool):
        raise TypeError("analytical KV capacity include_mtp must be boolean")
    if policy.pool_name != pool_name:
        raise ValueError(
            f"runtime memory policy pool {policy.pool_name!r} disagrees with "
            f"requested pool {pool_name!r}")
    _require_integer(
        "kv_block_size_tokens", kv_block_size_tokens, positive=True)
    _require_integer("max_model_len", max_model_len, positive=True)
    _require_integer("max_batched_tokens", max_batched_tokens, positive=True)
    _require_integer("max_requests", max_requests, positive=True)
    if max_model_len > model.max_position_embeddings:
        raise ValueError(
            "KV capacity max_model_len exceeds the canonical model limit")
    expected_binding = RuntimeMemoryPolicyBinding(
        hardware_id=hardware.hardware_id,
        # The policy records the source file digest for provenance, but a
        # serving simulation is selected by the live HWSpec, not YAML bytes.
        # Do not reject a current model solely because that provenance field
        # changed after policy capture.
        hardware_config_sha256=policy.binding.hardware_config_sha256,
        model_config_sha256=model.config_sha256,
        deployment=deployment,
        worker_role=worker_role,
        tensor_parallel_size=parallelism.tensor_parallel_size,
        pipeline_parallel_size=parallelism.pipeline_parallel_size,
        expert_parallel_size=parallelism.expert_parallel_size,
        max_model_len=max_model_len,
        max_batched_tokens=max_batched_tokens,
        max_requests=max_requests,
        kv_block_size_tokens=kv_block_size_tokens,
    )
    if policy.binding != expected_binding:
        raise ValueError(
            "runtime memory policy binding disagrees with analytical runtime")
    dram = hardware.resource_by_role("dram")
    physical_hbm_bytes = dram.size_bytes
    _require_integer("physical_hbm_bytes", physical_hbm_bytes, positive=True)
    utilized_hbm_bytes = policy.utilized_hbm_bytes(physical_hbm_bytes)
    residencies = model_rank_residencies(
        model,
        parallelism,
        worker_role=worker_role,
        include_mtp=include_mtp,
    )
    rank_budgets = tuple(
        KVCapacityRankBudget(
            rank=residency.rank,
            tensor_parallel_rank=residency.tensor_parallel_rank,
            pipeline_parallel_rank=residency.pipeline_parallel_rank,
            expert_parallel_rank=residency.expert_parallel_rank,
            layer_start=residency.layer_start,
            layer_end=residency.layer_end,
            physical_hbm_bytes=physical_hbm_bytes,
            utilized_hbm_bytes=utilized_hbm_bytes,
            model_weight_resident_bytes=residency.weight_resident_bytes,
            non_kv_reserve_bytes=policy.non_kv_reserve_bytes,
            available_kv_bytes=(
                utilized_hbm_bytes
                - residency.weight_resident_bytes
                - policy.non_kv_reserve_bytes),
        )
        for residency in residencies
    )
    allocation = allocator(
        model,
        parallelism,
        kv_block_size_tokens=kv_block_size_tokens,
        max_model_len=max_model_len,
        available_memory_by_rank=tuple(
            budget.available_kv_bytes for budget in rank_budgets),
    )
    if not isinstance(allocation, VLLMKVAllocation):
        raise TypeError("native KV allocator must return VLLMKVAllocation")
    if model.attention_layout == "standard_gqa":
        expected_spec_type = "FullAttentionSpec"
    elif model.attention_layout == "mla":
        expected_spec_type = "MLAAttentionSpec"
    elif model.attention_layout == "dsa_mla":
        expected_spec_type = "HeterogeneousMLAAttentionSpec"
    else:
        raise AssertionError(
            "model residency accepted an unsupported attention layout")
    if (allocation.kv_spec_type != expected_spec_type
            or allocation.kv_block_size_tokens != kv_block_size_tokens
            or allocation.max_model_len != max_model_len
            or len(allocation.worker_num_kv_blocks) != len(rank_budgets)
            or any(blocks != allocation.num_kv_blocks
                   for blocks in allocation.worker_num_kv_blocks)
            or len(allocation.worker_allocated_tensor_bytes)
            != len(rank_budgets)):
        raise RuntimeError(
            "native vLLM KV allocation disagrees with the analytical inputs")
    layer_ranges = pipeline_layer_ranges_for_model(
        model, parallelism.pipeline_parallel_size)
    stage_kv_bytes_per_token = tuple(
        model.kv_bytes_per_token_for_layer_range(
            parallelism.tensor_parallel_size,
            *layer_range,
        )
        for layer_range in layer_ranges
    )
    rank_kv_bytes_per_token = tuple(
        stage_kv_bytes_per_token[budget.pipeline_parallel_rank]
        for budget in rank_budgets
    )
    kv_bytes_per_token = max(rank_kv_bytes_per_token)
    kv_block_bytes = kv_block_size_tokens * kv_bytes_per_token
    if model.attention_layout != "dsa_mla":
        for budget, bytes_per_token in zip(
                rank_budgets, rank_kv_bytes_per_token, strict=True):
            local_layers = budget.layer_end - budget.layer_start
            native_bytes_per_block = (
                allocation.kv_page_size_bytes_per_layer * local_layers)
            if native_bytes_per_block != (
                    kv_block_size_tokens * bytes_per_token):
                raise RuntimeError(
                    "native vLLM KV spec bytes disagree with ModelConfig: "
                    f"native={native_bytes_per_block}, "
                    f"model={kv_block_size_tokens * bytes_per_token}")
    for allocated_bytes, bytes_per_token in zip(
            allocation.worker_allocated_tensor_bytes,
            rank_kv_bytes_per_token):
        if allocated_bytes < (
                allocation.num_kv_blocks
                * kv_block_size_tokens
                * bytes_per_token):
            raise RuntimeError(
                "native KV allocation cannot hold its rank-local Scheduler "
                "block count")
    return KVCapacityPlan(
        pool_name=pool_name,
        source_mode="analytical",
        kv_block_size_tokens=kv_block_size_tokens,
        kv_bytes_per_token=kv_bytes_per_token,
        kv_block_bytes=kv_block_bytes,
        num_kv_blocks=allocation.num_kv_blocks,
        hardware_id=hardware.hardware_id,
        model_config_sha256=model.config_sha256,
        tensor_parallel_size=parallelism.tensor_parallel_size,
        pipeline_parallel_size=parallelism.pipeline_parallel_size,
        expert_parallel_size=parallelism.expert_parallel_size,
        worker_role=worker_role,
        rank_budgets=rank_budgets,
        native_allocation=allocation,
    )


def resolve_debug_kv_capacity_plan(
    pool_name: str,
    *,
    kv_block_size_tokens: int,
    kv_bytes_per_token: int,
    num_kv_blocks: int,
) -> KVCapacityPlan:
    """Construct the explicitly named fixture/debug-only capacity path."""

    for name, value in (
            ("kv_block_size_tokens", kv_block_size_tokens),
            ("kv_bytes_per_token", kv_bytes_per_token),
            ("num_kv_blocks", num_kv_blocks)):
        _require_integer(name, value, positive=True)
    return KVCapacityPlan(
        pool_name=pool_name,
        source_mode="debug_explicit_block_count",
        kv_block_size_tokens=kv_block_size_tokens,
        kv_bytes_per_token=kv_bytes_per_token,
        kv_block_bytes=kv_block_size_tokens * kv_bytes_per_token,
        num_kv_blocks=num_kv_blocks,
        hardware_id=None,
        model_config_sha256=None,
        tensor_parallel_size=None,
        pipeline_parallel_size=None,
        expert_parallel_size=None,
        worker_role=None,
        rank_budgets=(),
        native_allocation=None,
    )


def kv_capacity_manifest(
    plans: Iterable[KVCapacityPlan],
    policy_config: Optional[RuntimeMemoryPolicyConfig],
) -> dict[str, object]:
    """Serialize request-safe analytical inputs, accounting, and allocation."""

    normalized = tuple(plans)
    if not normalized or any(
            not isinstance(plan, KVCapacityPlan) for plan in normalized):
        raise ValueError("KV capacity manifest requires resolved plans")
    pool_names = tuple(plan.pool_name for plan in normalized)
    if len(pool_names) != len(set(pool_names)):
        raise ValueError("KV capacity manifest pool names must be unique")
    modes = {plan.source_mode for plan in normalized}
    if policy_config is None:
        if modes != {"debug_explicit_block_count"}:
            raise ValueError(
                "KV capacity manifest without a runtime policy is debug-only")
        source_path = None
        source_sha256 = None
    elif isinstance(policy_config, RuntimeMemoryPolicyConfig):
        if modes != {"analytical"}:
            raise ValueError(
                "runtime memory policy requires analytical capacity plans")
        policy_config.validate_scheduler_pools(pool_names)
        source_path = policy_config.source_path
        source_sha256 = policy_config.source_sha256
    else:
        raise TypeError(
            "policy_config must be RuntimeMemoryPolicyConfig or null")
    entries = []
    for plan in sorted(normalized, key=lambda item: item.pool_name):
        policy = (
            None
            if policy_config is None
            else policy_config.policy_for_pool(plan.pool_name)
        )
        entries.append({
            "pool_name": plan.pool_name,
            "source_mode": plan.source_mode,
            "hardware_id": plan.hardware_id,
            "model_config_sha256": plan.model_config_sha256,
            "worker_role": plan.worker_role,
            "tensor_parallel_size": plan.tensor_parallel_size,
            "pipeline_parallel_size": plan.pipeline_parallel_size,
            "expert_parallel_size": plan.expert_parallel_size,
            "kv_block_size_tokens": plan.kv_block_size_tokens,
            "kv_bytes_per_token": plan.kv_bytes_per_token,
            "kv_block_bytes": plan.kv_block_bytes,
            "num_kv_blocks": plan.num_kv_blocks,
            "runtime_memory_policy": (
                None if policy is None else {
                    "binding": asdict(policy.binding),
                    "gpu_memory_utilization": {
                        "numerator": policy.gpu_memory_utilization_numerator,
                        "denominator": policy.gpu_memory_utilization_denominator,
                        "provenance": asdict(
                            policy.gpu_memory_utilization_provenance),
                    },
                    "non_kv_reserves": [
                        asdict(reserve) for reserve in policy.non_kv_reserves
                    ],
                    "non_kv_reserve_bytes": policy.non_kv_reserve_bytes,
                    "reserve_scope": policy.reserve_scope,
                    "completeness_assertion": policy.completeness_assertion,
                }
            ),
            "rank_budgets": [asdict(budget) for budget in plan.rank_budgets],
            "native_vllm_allocation": (
                None
                if plan.native_allocation is None
                else asdict(plan.native_allocation)
            ),
        })
    return {
        "schema_version": 2,
        "source": {
            "mode": next(iter(modes)),
            "runtime_memory_policy_path": source_path,
            "runtime_memory_policy_sha256": source_sha256,
        },
        "scheduler_pools": entries,
    }
