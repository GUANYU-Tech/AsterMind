"""Analytical HBM-to-KV capacity contract for production Serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Optional

from src.kernel.hardware_spec import HWSpec
from src.serving.kv_cache.model_residency import dsv4_model_buffer_bytes, model_rank_residencies
from src.serving.kv_cache.vllm_allocation import (
    VLLMKVAllocation,
    allocate_vllm_kv_cache,
)
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.workload.model_config import (
    ModelConfig, is_validated_qwen3_moe_ep2_model_scope, is_validated_qwen3_pipeline_model_scope,
)


_PROVENANCE_FIELDS = {"source_kind", "source_reference", "source_sha256"}
_POLICY_FIELDS = {
    "pool_name", "binding", "gpu_memory_utilization", "non_kv_reserves",
    "reserve_scope", "completeness_assertion",
}
_OPTIONAL_POLICY_FIELDS = {"runtime_visible_hbm"}
_VISIBLE_HBM_FIELDS = {"bytes", "provenance"}
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
_VERIFIED_RESERVE_SCOPE = "verified_uniform_across_pool_ranks"
_UNVERIFIED_RESERVE_SCOPE = "assumed_uniform_across_pool_ranks_unverified"
_VERIFIED_COMPLETENESS = "all_non_kv_reserves_for_bound_runtime"
_UNVERIFIED_COMPLETENESS = (
    "reserves_inherited_unverified_not_complete_for_bound_runtime")
_EVIDENCE_STATUS_BY_SCOPE = {
    (_VERIFIED_RESERVE_SCOPE, _VERIFIED_COMPLETENESS):
        "verified_runtime_evidence",
    (_UNVERIFIED_RESERVE_SCOPE, _UNVERIFIED_COMPLETENESS):
        "unverified_analytical_assumption",
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
    runtime_visible_hbm_bytes: Optional[int] = None
    runtime_visible_hbm_provenance: Optional[CapacityEvidence] = None

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
        if (self.runtime_visible_hbm_bytes is None) != (self.runtime_visible_hbm_provenance is None):
            raise ValueError("runtime visible HBM requires paired bytes and provenance")
        if self.runtime_visible_hbm_bytes is not None:
            _require_integer("runtime_visible_hbm bytes", self.runtime_visible_hbm_bytes, positive=True)
            if not isinstance(self.runtime_visible_hbm_provenance, CapacityEvidence):
                raise TypeError("runtime visible HBM provenance must be CapacityEvidence")
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
        if self.reserve_scope not in (
                _VERIFIED_RESERVE_SCOPE, _UNVERIFIED_RESERVE_SCOPE):
            raise ValueError(
                "runtime policy reserve_scope must be "
                f"{_VERIFIED_RESERVE_SCOPE!r} or {_UNVERIFIED_RESERVE_SCOPE!r}")
        if self.completeness_assertion not in (
                _VERIFIED_COMPLETENESS, _UNVERIFIED_COMPLETENESS):
            raise ValueError(
                "runtime policy completeness_assertion must be "
                f"{_VERIFIED_COMPLETENESS!r} or {_UNVERIFIED_COMPLETENESS!r}")
        if (self.reserve_scope, self.completeness_assertion) not in (
                _EVIDENCE_STATUS_BY_SCOPE):
            raise ValueError(
                "runtime policy reserve_scope and completeness_assertion "
                "must agree on the evidence status; mixed verified/unverified "
                "states fail closed")

    @property
    def evidence_status(self) -> str:
        """Evidence grade of this policy's values for the bound runtime."""

        return _EVIDENCE_STATUS_BY_SCOPE[
            (self.reserve_scope, self.completeness_assertion)]

    @property
    def non_kv_reserve_bytes(self) -> int:
        return sum(reserve.bytes for reserve in self.non_kv_reserves)

    @property
    def hbm_capacity_basis(self) -> str:
        """Absent runtime visibility is a nominal assumption, not a measurement."""
        return ("nominal_hardware_assumption" if self.runtime_visible_hbm_bytes is None
            else "explicit_runtime_visible_hbm")

    def utilized_hbm_bytes(self, physical_hbm_bytes: int) -> int:
        """Apply utilization to an explicit visible limit or nominal assumption.

        Runtime visibility limits capacity; it is not an allocated buffer or a
        reserve. Preserve the hardware's physical capacity separately.
        """
        _require_integer("physical_hbm_bytes", physical_hbm_bytes, positive=True)
        capacity = physical_hbm_bytes
        if self.runtime_visible_hbm_bytes is not None:
            if self.runtime_visible_hbm_bytes > physical_hbm_bytes:
                raise ValueError("runtime visible HBM cannot exceed physical HBM")
            capacity = self.runtime_visible_hbm_bytes
        return capacity * self.gpu_memory_utilization_numerator // self.gpu_memory_utilization_denominator


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
    """Resolved capacity consumed by one native framework scheduler.

    kv_block_bytes is a physical allocation row. In a native0.30 shared pool
    it need not equal the representative token block size times the logical
    average kv_bytes_per_token; complete native arena facts prove that row.
    """

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
    v4_weight_storage_profile: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pool_name, str) or not self.pool_name:
            raise ValueError("KV capacity plan pool_name must be non-empty")
        if self.source_mode not in (
                "analytical", "debug_explicit_block_count"):
            raise ValueError(
                f"unknown KV capacity source_mode: {self.source_mode!r}")
        if self.v4_weight_storage_profile is not None and (
                self.v4_weight_storage_profile != "sm90_deepgemm_marlin" or self.source_mode != "analytical"
                or self.hardware_id != "nvidia_h200" or self.native_allocation is None
                or self.native_allocation.version != "0.30.0"):
            raise ValueError("V4 storage profile requires analytical H200 native0.30 capacity")
        for name in (
                "kv_block_size_tokens", "kv_bytes_per_token",
                "kv_block_bytes", "num_kv_blocks"):
            _require_integer(name, getattr(self, name), positive=True)
        if self.kv_block_bytes != (
                self.kv_block_size_tokens * self.kv_bytes_per_token):
            allocation = self.native_allocation
            if (self.source_mode != "analytical" or allocation is None
                    or allocation.version != "0.30.0" or allocation.kv_spec_type != "UniformTypeKVCacheSpecs"
                    or allocation.num_kv_blocks != self.num_kv_blocks
                    or not allocation.worker_allocated_tensor_bytes
                    or any(size != self.num_kv_blocks * self.kv_block_bytes
                           for size in allocation.worker_allocated_tensor_bytes)):
                raise ValueError("KV capacity block bytes disagree with token geometry or native shared-pool arena")
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
        if not _POLICY_FIELDS <= set(entry) or set(entry) - _POLICY_FIELDS - _OPTIONAL_POLICY_FIELDS:
            raise ValueError(
                f"{label} fields require {sorted(_POLICY_FIELDS)}, "
                f"optional {sorted(_OPTIONAL_POLICY_FIELDS)}; got {sorted(entry)}")
        visible_bytes = visible_provenance = None
        if "runtime_visible_hbm" in entry:
            visible = entry["runtime_visible_hbm"]
            if not isinstance(visible, dict):
                raise TypeError(f"{label}.runtime_visible_hbm must be a mapping")
            if set(visible) != _VISIBLE_HBM_FIELDS:
                raise ValueError(f"{label}.runtime_visible_hbm fields must be exactly {sorted(_VISIBLE_HBM_FIELDS)}")
            visible_bytes = _require_integer("runtime_visible_hbm bytes", visible["bytes"], positive=True)
            visible_provenance = _load_provenance(visible["provenance"], f"{label}.runtime_visible_hbm.provenance")
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
            runtime_visible_hbm_bytes=visible_bytes, runtime_visible_hbm_provenance=visible_provenance,
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


def runtime_memory_policy_payload(policies: tuple[RuntimeMemoryPolicy, ...]) -> dict[str, object]:
    """Serialize the existing policy schema, including generated candidates."""
    if not isinstance(policies, tuple) or not policies or any(not isinstance(p, RuntimeMemoryPolicy) for p in policies):
        raise TypeError("runtime policy payload requires explicit RuntimeMemoryPolicy objects")
    if len({p.pool_name for p in policies}) != len(policies):
        raise ValueError("runtime policy payload cannot repeat a scheduler pool")
    return {"schema_version": 2, "scheduler_pools": [{
        "pool_name": policy.pool_name, "binding": asdict(policy.binding),
        "gpu_memory_utilization": {"numerator": policy.gpu_memory_utilization_numerator,
            "denominator": policy.gpu_memory_utilization_denominator,
            "provenance": asdict(policy.gpu_memory_utilization_provenance)},
        "non_kv_reserves": [asdict(reserve) for reserve in policy.non_kv_reserves],
        "reserve_scope": policy.reserve_scope, "completeness_assertion": policy.completeness_assertion,
        **({"runtime_visible_hbm": {"bytes": policy.runtime_visible_hbm_bytes,
            "provenance": asdict(policy.runtime_visible_hbm_provenance)}}
            if policy.runtime_visible_hbm_bytes is not None else {}),
    } for policy in policies]}


def project_native_model_loading_policy(
    policy: RuntimeMemoryPolicy, *, pure_parameter_bytes: int,
    known_constructor_buffer_bytes: int,
    observed_model_loading_allocated_increase_bytes: int,
    observed_non_torch_increase_bytes: int,
    framework_coverage: str,
    model_loading_provenance: CapacityEvidence,
    non_torch_provenance: CapacityEvidence,
) -> tuple[RuntimeMemoryPolicy, dict[str, object]]:
    """Charge measured nonparameter loading bytes once in framework.

    The base framework must explicitly represent only the supplied native
    non-Torch counter. Loading covers parameters plus constructor buffers and
    possibly other allocations: its unexplained remainder stays unnamed but
    is still part of that measured increment. No KV geometry, parameter field
    or other reserve changes, and source constructor bytes are not added twice.

    The combined framework_source provenance describes the static accounting
    rule N + (L - P), not the origin of the measured byte values. Both original
    native runtime counter references are retained in that provenance and the
    diagnostic. Its distinct kind also prevents projecting a composed reserve
    again as though it were the original non-Torch-only runtime counter.
    """
    if not isinstance(policy, RuntimeMemoryPolicy):
        raise TypeError("native loading projection requires RuntimeMemoryPolicy")
    if framework_coverage != "non_torch_only":
        raise ValueError("native loading projection requires explicit non_torch_only framework coverage")
    for name, value, positive in (
            ("pure_parameter_bytes", pure_parameter_bytes, True),
            ("known_constructor_buffer_bytes", known_constructor_buffer_bytes, False),
            ("observed_model_loading_allocated_increase_bytes", observed_model_loading_allocated_increase_bytes, True),
            ("observed_non_torch_increase_bytes", observed_non_torch_increase_bytes, False)):
        _require_integer(name, value, positive=positive)
    for evidence in (model_loading_provenance, non_torch_provenance):
        if not isinstance(evidence, CapacityEvidence):
            raise TypeError("native counter provenance requires CapacityEvidence")
        if evidence.source_kind != "runtime_configuration":
            raise ValueError("native counter provenance must identify actual runtime configuration evidence")
    framework = next(item for item in policy.non_kv_reserves if item.kind == "framework")
    if (framework.provenance.source_kind != "runtime_configuration"
            or framework.bytes != observed_non_torch_increase_bytes):
        raise ValueError("base framework must contain only the supplied native non-Torch counter; opaque or composed reserves cannot be combined")
    nonparameter = observed_model_loading_allocated_increase_bytes - pure_parameter_bytes
    if nonparameter < known_constructor_buffer_bytes:
        raise ValueError("observed model-loading increment is below parameters plus known constructor buffers")
    unassigned = nonparameter - known_constructor_buffer_bytes
    evidence = {"base_framework": asdict(framework.provenance),
        "model_loading_counter": asdict(model_loading_provenance),
        "non_torch_counter": asdict(non_torch_provenance)}
    composed = RuntimeMemoryReserve("framework", observed_non_torch_increase_bytes + nonparameter,
        CapacityEvidence("framework_source",
            "native_non_torch_plus_model_loading_nonparameter; N=" + str(observed_non_torch_increase_bytes)
            + "; L=" + str(observed_model_loading_allocated_increase_bytes) + "; P=" + str(pure_parameter_bytes)
            + "; constructor_is_subset=" + str(known_constructor_buffer_bytes)
            + "; unassigned_loading_subset=" + str(unassigned)
            + "; provenance=" + json.dumps(evidence, sort_keys=True), None))
    candidate = replace(policy,
        non_kv_reserves=tuple(composed if item.kind == "framework" else item for item in policy.non_kv_reserves),
        reserve_scope=_UNVERIFIED_RESERVE_SCOPE, completeness_assertion=_UNVERIFIED_COMPLETENESS)
    return candidate, {"framework_coverage": "non_torch_plus_observed_model_nonparameter_loading",
        "pure_parameter_bytes": pure_parameter_bytes,
        "known_constructor_buffer_bytes": known_constructor_buffer_bytes,
        "observed_model_loading_allocated_increase_bytes": observed_model_loading_allocated_increase_bytes,
        "observed_non_torch_increase_bytes": observed_non_torch_increase_bytes,
        "model_loading_nonparameter_bytes": nonparameter,
        "unassigned_model_loading_increment_bytes": unassigned,
        "framework_reserve_bytes": composed.bytes, "provenance": evidence,
        "normal_capacity_accepted": False,
        "scope": "measured loading increment charged once; runtime/KV identity and remaining startup coverage independently required"}


def _project_source_storage_reserves(policy, prediction, *, framework_source_epoch):
    """Partition the actual source allocations; never subtract traffic or views.

    The source budget charges final resident objects in framework once and
    simultaneous nonretained objects in activation. It covers this declared
    source program only, not unknown runtime/workspace or warmup programs.
    """
    if framework_source_epoch not in ("before_profile", "after_profile"):
        raise ValueError("source policy requires explicit before_profile/after_profile framework epoch")
    records = tuple(row["profile_storage_measurements"] for row in prediction["ranks"].values())
    framework = next(reserve for reserve in policy.non_kv_reserves if reserve.kind == "framework")
    epoch_field = "before_profile_source_bytes" if framework_source_epoch == "before_profile" else "after_profile_source_bytes"
    if framework.bytes < max(record[epoch_field] for record in records):
        raise ValueError("framework reserve is below source-defined retained objects at its declared epoch")
    if framework_source_epoch == "before_profile":
        if any(record["retained_source_growth_bytes"] < 0 for record in records):
            raise NotImplementedError("source framework projection does not infer how external reserves release with negative growth")
        growth = max(record["retained_source_growth_bytes"] for record in records)
        framework = RuntimeMemoryReserve("framework", framework.bytes + growth,
            CapacityEvidence("framework_source", "base framework before_profile: " + framework.provenance.source_reference
                + "; add maximum per-rank source retained growth=" + str(growth)
                + "B once; inherited external reserve scope remains unverified", None))
    activation_bytes = max(record["nonretained_source_peak_bytes"] for record in records)
    for record in records:
        if (framework.bytes < record["after_profile_source_bytes"]
                or framework.bytes + activation_bytes < record["peak_source_storage_bytes"]):
            raise AssertionError("projected source ledger does not cover a physical rank's source peak")
    prediction["activation_policy_budget"] = {"bytes": activation_bytes,
        "framework_bytes": framework.bytes, "framework_source_epoch": framework_source_epoch,
        "semantics": "source_budget_bound_final_resident_once_plus_peak_nonretained_storage",
        "observed_point_estimate": False, "normal_capacity_accepted": False}
    activation = RuntimeMemoryReserve("activation", activation_bytes, CapacityEvidence("framework_source",
        "symbolic startup source_program=" + prediction["source_program"]
        + ("; model_execution_version=" + prediction["model_execution_version"] if "model_execution_version" in prediction else "")
        + "; maximum per-rank simultaneous nonretained source allocation bytes; "
        "final resident allocations charged in framework only; source inputs and witness in the source report; "
        "not an observed point estimate; library/compiler/runtime completeness and inherited external reserves unverified", None))
    projected = replace(policy,
        non_kv_reserves=tuple(activation if item.kind == "activation" else framework if item.kind == "framework" else item
            for item in policy.non_kv_reserves), reserve_scope=_UNVERIFIED_RESERVE_SCOPE,
        completeness_assertion=_UNVERIFIED_COMPLETENESS)
    return projected, prediction


def project_symbolic_activation_policy(policy: RuntimeMemoryPolicy, *, model: ModelConfig,
        parallelism: ParallelConfig, hardware: HWSpec, source_profile: str, framework_source_epoch=None,
        source_use_deep_gemm=None, workspace_initial_capacity_bytes=None, indexer_max_logits_mb=None, expert_routing=None,
        collective_configs=(), mtp_num_speculative_tokens=0, model_execution_version="vllm_0_30_0",
        source_lifetime_policy=None, source_compilation_options=None):
    """Replace source storage reserves with an explicit, unverified projection.

    Other reserves remain external inputs inherited from the supplied policy;
    an explicitly before-profile framework reserve gains source retained
    growth once. An after-profile reserve already owns that storage.
    This incomplete source projection always downgrades their combined evidence
    grade. A framework reserve below the represented retained state is rejected;
    passing that lower bound does not prove workspace/runtime completeness.
    No observed activation value influences the new source-derived byte count.
    Final residency plus maximum nonretained source storage bounds the
    declared startup program; it does not bound unknown runtime programs.
    The dense CUDA source is shared by declared A800/H200 scenarios; accepting
    that source does not rebind an inherited 0.11 policy to a 0.30 runtime.
    The V4 storage program retains its separate explicit H200 restriction.
    GLM target-only or two-pass MTP storage shares this projection only;
    it does not establish normal residency or native allocator support.
    """
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
        raise NotImplementedError("startup policy requires its version-bound source")
    flashinfer011 = source_profile == "cuda_eager_inplace_fa_no_metadata_flashinfer_0_4_1_top_k_first"
    ascend018 = source_profile == "vllm_ascend_0_18_no_metadata_head_only"
    metax024 = source_profile == "vllm_metax_0_24_no_metadata_native_triton_fp32_noise_cold_cache"
    from src.serving.kv_cache.activation_memory import (
        symbolic_startup_activation_peak, symbolic_v4_startup_activation_peak, symbolic_glm_startup_activation_peak,
    )
    if not isinstance(policy, RuntimeMemoryPolicy) or not isinstance(model, ModelConfig):
        raise TypeError("symbolic activation projection requires normalized policy and model")
    if model_execution_version != "vllm_0_30_0" and model.model_type not in ("qwen3", "llama"):
        raise NotImplementedError("startup0.11 source is limited to ordinary dense models")
    if not isinstance(parallelism, ParallelConfig) or not isinstance(hardware, HWSpec):
        raise TypeError("symbolic activation projection requires normalized parallelism and hardware")
    if flashinfer011 and (model_execution_version != "vllm_0_11_0" or hardware.hardware_id != "nvidia_a800"):
        raise NotImplementedError("FlashInfer0.4.1 startup policy requires explicit A8000.11 source")
    if metax024:
        if (model_execution_version != "vllm_0_24_0" or hardware.hardware_id != "metax_c500"
                or parallelism.tensor_parallel_size != 1 or parallelism.pipeline_parallel_size != 1
                or parallelism.enable_expert_parallel):
            raise NotImplementedError("MetaX startup policy requires explicit C5000.24 BF16 dense TP1 source")
    elif model_execution_version == "vllm_0_24_0":
        raise ValueError("MetaX0.24 startup cannot borrow another version's sampler profile")
    if ascend018:
        if (model_execution_version != "vllm_0_18_0" or hardware.hardware_id != "ascend_910b"
                or model.model_type != "qwen3" or model.compute_dtype != model.weight_dtype or model.compute_dtype != "bf16"
                or parallelism.tensor_parallel_size != 1 or parallelism.pipeline_parallel_size != 1
                or parallelism.enable_expert_parallel):
            raise NotImplementedError("Ascend startup policy requires explicit Qwen3 BF16 TP1 Ascend0.18 source")
    elif model_execution_version == "vllm_0_18_0" or source_lifetime_policy is not None or source_compilation_options is not None:
        raise ValueError("source lifetime/compilation options require the explicit Ascend0.18 startup source")
    if type(mtp_num_speculative_tokens) is not int or mtp_num_speculative_tokens not in (0, 2):
        raise ValueError("startup source requires explicit zero or two speculative tokens")
    glm_source = source_profile in ("vllm_0_30_v32_bf16_sync_explicit_allreduce",
        "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    if not glm_source and mtp_num_speculative_tokens:
        raise ValueError("startup speculative passes require the GLM source")
    binding = policy.binding
    if ((hardware.hardware_id not in ("nvidia_a800", "nvidia_h200") and not (ascend018 or metax024)) or binding.hardware_id != hardware.hardware_id
            or binding.model_config_sha256 != model.config_sha256
            or (binding.tensor_parallel_size, binding.pipeline_parallel_size, binding.expert_parallel_size)
            != (parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size, parallelism.expert_parallel_size)
            or binding.deployment != "colocated" or binding.worker_role != "colocated"
            or model.config_extraction_path != "$"):
        raise ValueError("symbolic activation policy binding disagrees with the supported source")
    graph = next(reserve for reserve in policy.non_kv_reserves if reserve.kind == "cuda_graph")
    # The explicit0.11 worker profiles before capture. Its eager source
    # storage can coexist with an independently supplied graph reserve;
    # inheriting that reserve does not derive capture or warmup memory.
    if graph.bytes != 0 and not (flashinfer011 or ascend018 or metax024):
        raise NotImplementedError("eager source activation cannot inherit a nonzero CUDA graph reserve")
    if glm_source:
        if collective_configs or source_use_deep_gemm is not None:
            raise ValueError("GLM startup owns its collective declaration and does not consume V4 branches")
        if framework_source_epoch not in ("before_profile", "after_profile"):
            raise ValueError("GLM policy requires an explicit framework source epoch; no complete runtime policy is inferred")
        if any(value is None for value in (workspace_initial_capacity_bytes, indexer_max_logits_mb, expert_routing)):
            raise ValueError("GLM policy requires explicit workspace history, logits MiB and routing")
        prediction = symbolic_glm_startup_activation_peak(model, parallelism, hardware=hardware,
            pool_name=policy.pool_name, tokens=binding.max_batched_tokens, max_requests=binding.max_requests,
            source_max_model_len=binding.max_model_len, source_profile=source_profile,
            workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
            indexer_max_logits_mb=indexer_max_logits_mb, expert_routing=expert_routing,
            include_mtp=mtp_num_speculative_tokens > 0, num_speculative_tokens=mtp_num_speculative_tokens)
        return _project_source_storage_reserves(policy, prediction, framework_source_epoch=framework_source_epoch)
    if source_profile == "vllm_0_30_v4_sync_no_metadata_pytorch_sort_fp32_noise":
        if collective_configs:
            raise ValueError("V4 startup source owns its separate collective storage declaration")
        if hardware.hardware_id != "nvidia_h200":
            raise NotImplementedError("V4 startup storage source requires its explicit H200 branch")
        if framework_source_epoch not in ("before_profile", "after_profile"):
            raise ValueError("V4 policy requires an explicit framework source epoch; no complete runtime policy is inferred")
        if any(value is None for value in (source_use_deep_gemm, workspace_initial_capacity_bytes,
                indexer_max_logits_mb, expert_routing)):
            raise ValueError("V4 policy requires explicit source branch, workspace history, logits MiB and routing")
        known_buffers = sum(dsv4_model_buffer_bytes(model, parallelism, max_batched_tokens=binding.max_batched_tokens).values())
        framework = next(reserve for reserve in policy.non_kv_reserves if reserve.kind == "framework")
        if framework.bytes < known_buffers:
            raise ValueError("V4 framework reserve is below source-defined persistent model buffers")
        prediction = symbolic_v4_startup_activation_peak(model, parallelism, hardware=hardware, pool_name=policy.pool_name,
            tokens=binding.max_batched_tokens, max_requests=binding.max_requests, source_max_model_len=binding.max_model_len,
            source_profile=source_profile, source_use_deep_gemm=source_use_deep_gemm,
            workspace_initial_capacity_bytes=workspace_initial_capacity_bytes, indexer_max_logits_mb=indexer_max_logits_mb,
            expert_routing=expert_routing)
        return _project_source_storage_reserves(policy, prediction, framework_source_epoch=framework_source_epoch)
    if model.num_mtp_layers or any(value is not None for value in (source_use_deep_gemm,
            workspace_initial_capacity_bytes, indexer_max_logits_mb, expert_routing)):
        raise ValueError("dense activation source does not consume V4 inputs or MTP")
    fp16_sort_source = (model.model_type == "qwen3" and model.compute_dtype == model.weight_dtype == "fp16"
        and source_profile == "cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise")
    if (model.model_type not in ("qwen3", "llama") or model.num_experts
            or not (model.compute_dtype == model.weight_dtype == "bf16" or fp16_sort_source)):
        raise NotImplementedError("dense activation source requires unquantized BF16 Qwen3/Llama or explicit FP16 Qwen3 sort")
    if framework_source_epoch not in (None, "after_profile"):
        raise ValueError("dense activation source requires the existing after_profile framework contract")
    if source_profile == "cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise" or flashinfer011 or ascend018:
        units = None
    elif source_profile == "cuda_eager_inplace_fa_no_metadata_native_triton_fp32_noise_cold_cache" or metax024:
        units = hardware.num_compute_units
    else:
        raise NotImplementedError("analytical activation needs an explicit supported source profile")
    prediction = symbolic_startup_activation_peak(model, parallelism,
        tokens=binding.max_batched_tokens, max_requests=binding.max_requests,
        source_profile=source_profile, num_compute_units=units, collective_configs=collective_configs,
        model_execution_version=model_execution_version,
        source_lifetime_policy=source_lifetime_policy, source_compilation_options=source_compilation_options)
    if flashinfer011 or ascend018 or metax024:
        prediction["profile_budget_scope"] = {
            "activation_source": "explicit_segment_profile_only" if source_lifetime_policy is not None else "eager_profile_only",
            "source_lifetime_policy": source_lifetime_policy,
            "source_compilation_options": source_compilation_options,
            "framework_source_epoch": "after_profile",
            "source_resident_storage_required_in_framework": True,
            "native_profile_budget_equivalence": False,
            "inherited_cuda_graph_reserve_bytes": graph.bytes,
            "cuda_graph_reserve_derived": False,
            "capture_and_warmup_memory_predicted": False,
        }
    if metax024:
        prediction["sampling_source_conditions"] = {"platform": "metax_out_of_tree", "HAS_TRITON": True,
            "logprobs_mode": "raw_logprobs", "use_fp64_gumbel": False, "sampler_calls": 2,
            "filter": "PyTorch_inplace_sort_below8_rows_otherwise0.24_Triton_no_split_workspace",
            "initial_cache": "cold", "actual_runtime_branch_observed": False}
    return _project_source_storage_reserves(policy, prediction, framework_source_epoch="after_profile")


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
    scheduler_version: str = "0.11.0",
    v4_weight_storage_profile: str | None = None,
) -> KVCapacityPlan:
    """Derive rank budgets and require the selected version's native facts.

    The 0.30 SDK allocator must be supplied explicitly by its isolated owner;
    the legacy default cannot establish 0.30 capacity. A matching receipt does
    not establish complete runtime reserve or provider provenance.
    """

    if worker_role == "ffn":
        raise NotImplementedError("AF FFN weight residency is non-KV; no Scheduler KV plan is defined")
    if not isinstance(pool_name, str) or not pool_name:
        raise ValueError("KV capacity pool_name must be non-empty")
    if not isinstance(hardware, HWSpec):
        raise TypeError("analytical KV capacity requires HWSpec")
    if not isinstance(model, ModelConfig):
        raise TypeError("analytical KV capacity requires ModelConfig")
    if model.config_extraction_path != "$":
        raise NotImplementedError(
            "normal KV capacity cannot use an extracted text_config as "
            "complete model residency: outer multimodal stages and the "
            "actual language_model_only/vision_chunk skip, loader or "
            "offload binding are unsupported")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("analytical KV capacity requires ParallelConfig")
    if not isinstance(policy, RuntimeMemoryPolicy):
        raise TypeError("analytical KV capacity requires RuntimeMemoryPolicy")
    if not isinstance(include_mtp, bool):
        raise TypeError("analytical KV capacity include_mtp must be boolean")
    if scheduler_version not in ("0.11.0", "0.30.0"):
        raise ValueError("unsupported analytical KV Scheduler version")
    if scheduler_version == "0.30.0":
        if allocator is allocate_vllm_kv_cache:
            raise NotImplementedError("0.30 analytical KV capacity requires an explicit native 0.30 allocator")
        if include_mtp:
            raise NotImplementedError("0.30 MTP normal capacity requires a complete startup activation ledger and runtime reserves; the conditional constructor-sized GLM target/drafter source program does not establish all-routing workspace bounds, sparse-index-group/speculative metadata, library/compiler state or allocator reservation")
        source_pp = (deployment == "colocated" and worker_role == "colocated" and hardware.hardware_id == "nvidia_a800"
            and not parallelism.enable_expert_parallel
            and is_validated_qwen3_pipeline_model_scope(model, parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size))
        if (model.model_type not in ("glm_moe_dsa", "deepseek_v4", "qwen3", "qwen3_moe", "llama")
                or (parallelism.pipeline_parallel_size != 1 and not source_pp)
                or (parallelism.enable_expert_parallel and not (
                    parallelism.tensor_parallel_size == 2 and is_validated_qwen3_moe_ep2_model_scope(model)))):
            raise NotImplementedError("0.30 analytical KV facts require ordinary GLM/V4/Qwen3/Llama PP1/MTP-off; EP is limited to validated Qwen3MoE TP=EP=2")
    if model.model_type == "deepseek_v4":
        if (scheduler_version != "0.30.0" or hardware.hardware_id != "nvidia_h200"
                or v4_weight_storage_profile != "sm90_deepgemm_marlin" or worker_role != "colocated"):
            raise NotImplementedError("normal V4 source capacity requires explicit H200/0.30 SM90 DeepGEMM/Marlin storage")
        # The existing policy's framework reserve must include these model
        # buffers once. Do not silently subtract them a second time or omit
        # them because Serving uses a shorter sequence limit than the config.
        buffers = dsv4_model_buffer_bytes(model, parallelism, max_batched_tokens=max_batched_tokens)
        framework = next(reserve for reserve in policy.non_kv_reserves if reserve.kind == "framework")
        if framework.bytes < sum(buffers.values()):
            raise ValueError("V4 framework reserve is smaller than source-defined persistent model buffers")
    elif v4_weight_storage_profile is not None:
        raise ValueError("V4 weight storage profile requires a V4 model")
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
    residency_options = {"worker_role": worker_role, "include_mtp": include_mtp}
    if scheduler_version == "0.30.0":
        residency_options["model_execution_version"] = "vllm_0_30_0"
    if v4_weight_storage_profile is not None:
        residency_options["v4_weight_storage_profile"] = v4_weight_storage_profile
    residencies = model_rank_residencies(model, parallelism, **residency_options)
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
    if allocation.version != scheduler_version:
        raise RuntimeError("native KV allocation version disagrees with the selected Scheduler")
    if scheduler_version == "0.30.0":
        expected_spec_type = "FullAttentionSpec" if model.attention_layout == "standard_gqa" else "UniformTypeKVCacheSpecs"
    elif model.attention_layout == "standard_gqa":
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
    if model.model_type == "deepseek_v4":
        # PP1 is enforced above. This average is informational; physical
        # storage comes from the native mixed-group arena below.
        stage_kv_bytes_per_token = (model.kv_bytes_per_token_per_parallel_rank(parallelism.tensor_parallel_size, 1),)
    else:
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
    logical_rate = max(rank_kv_bytes_per_token)
    kv_bytes_per_token = -(-logical_rate.numerator // logical_rate.denominator) if model.model_type == "deepseek_v4" else logical_rate
    kv_block_bytes = kv_block_size_tokens * kv_bytes_per_token
    if model.model_type == "deepseek_v4":
        row_sizes = tuple(divmod(size, allocation.num_kv_blocks) for size in allocation.worker_allocated_tensor_bytes)
        if any(remainder or row != row_sizes[0][0] for row, remainder in row_sizes):
            raise RuntimeError("V4 shared cache arenas must contain identical integral pool rows")
        kv_block_bytes = row_sizes[0][0]
        if any(size > budget.available_kv_bytes for size, budget in zip(
                allocation.worker_allocated_tensor_bytes, rank_budgets, strict=True)):
            raise RuntimeError("V4 native shared cache arena exceeds its rank-local budget")
    elif model.attention_layout != "dsa_mla":
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
        if model.model_type == "deepseek_v4":
            # Native mixed groups have different logical token block sizes;
            # the full arena/row validation above owns physical capacity.
            continue
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
        v4_weight_storage_profile=v4_weight_storage_profile,
    )


def resolve_symbolic_kv_capacity_plan(
    policy: RuntimeMemoryPolicy,
    *,
    model: ModelConfig,
    parallelism: ParallelConfig,
    hardware: HWSpec,
    hardware_config_sha256: str,
    source_profile: str,
    scheduler_version: str,
    allocator: Callable[..., VLLMKVAllocation],
    framework_source_epoch=None,
    source_use_deep_gemm=None,
    workspace_initial_capacity_bytes=None,
    indexer_max_logits_mb=None,
    expert_routing=None,
    v4_weight_storage_profile=None,
    collective_configs=(),
    model_execution_version="vllm_0_30_0",
    source_lifetime_policy=None,
    source_compilation_options=None,
) -> tuple[RuntimeMemoryPolicy, KVCapacityPlan, dict[str, object]]:
    """Use traced startup lifetimes in the ordinary physical KV budget.

    Constructor limits come only from the bound policy. No debug block count
    or observed activation bytes enter the source calculation. External
    non-activation policy values remain explicit unverified budget inputs;
    this function does not derive them from a startup accounting remainder.
    The returned native allocation is usable by the existing CPU Scheduler,
    while missing complete runtime reserves still reject normal acceptance.
    """
    projected, prediction = project_symbolic_activation_policy(policy, model=model,
        parallelism=parallelism, hardware=hardware, source_profile=source_profile,
        framework_source_epoch=framework_source_epoch, source_use_deep_gemm=source_use_deep_gemm,
        workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
        indexer_max_logits_mb=indexer_max_logits_mb, expert_routing=expert_routing,
        collective_configs=collective_configs, model_execution_version=model_execution_version,
        source_lifetime_policy=source_lifetime_policy, source_compilation_options=source_compilation_options)
    binding = projected.binding
    plan = resolve_analytical_kv_capacity_plan(projected.pool_name, hardware=hardware,
        hardware_config_sha256=hardware_config_sha256, model=model, parallelism=parallelism,
        deployment=binding.deployment, worker_role=binding.worker_role,
        kv_block_size_tokens=binding.kv_block_size_tokens, max_model_len=binding.max_model_len,
        max_batched_tokens=binding.max_batched_tokens, max_requests=binding.max_requests,
        policy=projected, allocator=allocator, scheduler_version=scheduler_version,
        v4_weight_storage_profile=v4_weight_storage_profile)
    external = tuple({"kind": reserve.kind, "bytes": reserve.bytes,
        "provenance": asdict(reserve.provenance), "analytically_derived": False,
        "runtime_binding_verified": False} for reserve in policy.non_kv_reserves if reserve.kind != "activation")
    report = {"source": "symbolic_startup_lifetimes_to_native_kv_allocation",
        "capacity_mode": plan.source_mode, "scheduler_version": scheduler_version,
        "model_execution_version": model_execution_version,
        "startup_prediction": prediction, "activation_budget": prediction["activation_policy_budget"],
        "weight_resident_bytes_by_rank": {str(rank.rank): rank.model_weight_resident_bytes for rank in plan.rank_budgets},
        "available_kv_bytes_by_rank": {str(rank.rank): rank.available_kv_bytes for rank in plan.rank_budgets},
        "native_kv_blocks": plan.num_kv_blocks, "non_activation_policy_inputs": external,
        "unknown_source_memory_categories": tuple(prediction["uncovered"]),
        "normal_capacity_accepted": False,
        "normal_capacity_rejection_reasons": (
            "inherited non-activation policy inputs are not a complete independently bound runtime ledger",
            "source startup allocation lifetimes do not establish the uncovered runtime memory categories"),
        "startup_log_comparison": None,
        "startup_log_comparison_unavailable_reason": "no matching source-profile and constructor-bound startup observation supplied"}
    return projected, plan, report


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


def validate_ffn_source_capacity(source_memory, *, model, deployment, hardware_by_id):
    """Reject impossible known FFN storage; never infer absent runtime reserves."""
    from src.serving.deployment.deployment_config import worker_role_operator_group
    if not isinstance(source_memory, Mapping) or not isinstance(model, ModelConfig):
        raise TypeError("FFN source capacity requires canonical model and source memory")
    ffns = tuple(pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "ffn")
    if not ffns:
        return {"pools": {}, "normal_capacity_accepted": False}
    reported, total = source_memory["reported_steps"], source_memory["total_simulation_steps"]
    all_reported = source_memory["all_steps_reported"]
    if (type(reported) is not int or type(total) is not int or total <= 0 or not 0 <= reported <= total
            or type(all_reported) is not bool or all_reported != (reported == total)
            or source_memory["complete"] is not False or source_memory["normal_capacity_accepted"] is not False):
        raise ValueError("FFN source capacity requires honest observed-step coverage without normal acceptance")
    pools = {}
    for pool in ffns:
        if pool.name not in source_memory["pools"]:
            pools[pool.name] = {"status": "unavailable_unexecuted_ffn_source", "ranks": {},
                "normal_capacity_accepted": False}
            continue
        hardware = hardware_by_id[pool.hardware_id]
        if not isinstance(hardware, HWSpec) or hardware.hardware_id != pool.hardware_id:
            raise ValueError("FFN source capacity requires its actual pool HWSpec")
        capacity = hardware.resource_by_role("dram").size_bytes
        _require_integer("FFN physical_hbm_bytes", capacity, positive=True)
        records = source_memory["pools"][pool.name]["ranks"]
        if (not isinstance(records, Mapping) or set(records) != {str(rank) for rank in pool.ranks}
                or len(set(pool.ranks)) != len(pool.ranks)):
            raise ValueError("FFN source capacity requires every physical rank exactly")
        peaks = {rank: _validated_source_peak_witness(record) for rank, record in records.items()}
        # Unsupported inventories remain explicit; no zero/reserve surrogate.
        try:
            residency = model_rank_residencies(model, pool.parallel_plan.config, worker_role="ffn")
        except NotImplementedError as error:
            pools[pool.name] = {"status": "unsupported_ffn_weight_residency", "reason": str(error),
                "ranks": {rank: {"known_activation_peak_bytes": peak, "witness_step_id": step,
                    "model_weight_resident_bytes": None, "runtime_reserves_bytes": None}
                    for rank, (peak, step) in peaks.items()}, "normal_capacity_accepted": False}
            continue
        if len(residency) != len(pool.ranks):
            raise ValueError("FFN source residency disagrees with physical placement")
        rows = {}
        for local, weights in enumerate(residency):
            rank = pool.ranks[local]
            peak, step_id = peaks[str(rank)]
            lower_bound = weights.weight_resident_bytes + peak
            if lower_bound > capacity:
                raise ValueError(f"known FFN weights plus source peak exceed physical HBM: pool={pool.name} "
                    f"rank={rank} step={step_id} known={lower_bound} capacity={capacity}")
            rows[str(rank)] = {"capacity_local_rank": local,
                "model_weight_resident_bytes": weights.weight_resident_bytes,
                "known_activation_peak_bytes": peak, "witness_step_id": step_id,
                "physical_hbm_bytes": capacity, "known_weight_plus_source_peak_bytes": lower_bound,
                "physical_hbm_lower_bound_margin_bytes": capacity - lower_bound,
                "runtime_reserves_bytes": None, "normal_capacity_accepted": False}
        pools[pool.name] = {"status": "known_source_lower_bound_fits_physical_hbm_unknown_reserves",
            "ranks": rows, "normal_capacity_accepted": False}
    return {"pools": pools, "normal_capacity_accepted": False,
        "reported_steps": reported, "total_simulation_steps": total, "all_steps_reported": all_reported,
        "scope": "ffn_weights_plus_observed_source_lower_bound_not_complete_worker_budget",
        "limitations": ("runtime_utilization_and_reserves_not_supplied", "no_KV_plan_for_FFN",
            "executed_request_peak_is_not_startup_or_all_workloads_bound")}


def _validated_source_peak_witness(record):
    peak, witness, step_id = record["known_activation_peak_bytes"], record["peak_live_storage"], record["step_id"]
    if (type(peak) is not int or peak < 0 or not isinstance(step_id, str) or not step_id
            or not isinstance(witness, (tuple, list))
            or any(not isinstance(item, Mapping)
                or not isinstance(item["allocation_id"], str) or not item["allocation_id"]
                or type(item["size_bytes"]) is not int or item["size_bytes"] <= 0 for item in witness)
            or len({item["allocation_id"] for item in witness}) != len(witness)
            or sum(item["size_bytes"] for item in witness) != peak):
        raise ValueError("source capacity peak requires its unique live-storage witness")
    return peak, step_id


def validate_source_activation_capacity(
    source_memory: Mapping[str, object],
    *,
    plans: Iterable[KVCapacityPlan],
    policy_config: RuntimeMemoryPolicyConfig,
    pool_ranks: Mapping[str, tuple[int, ...]],
) -> dict[str, object]:
    """Require executed source peaks to fit the existing normal rank ledger.

    This does not size startup memory, replace any reserve, or establish
    compiled/runtime completeness. Native KV backing bytes and all existing
    non-activation reserves stay charged. Pool-local capacity ranks map to
    actual trace ranks through the deployment's explicit placement.
    """
    if not isinstance(source_memory, Mapping):
        raise TypeError("normal source capacity requires the actual source memory report")
    if not isinstance(policy_config, RuntimeMemoryPolicyConfig):
        raise TypeError("normal source capacity requires a bound runtime policy")
    normalized = tuple(plans)
    if not normalized or any(not isinstance(plan, KVCapacityPlan) for plan in normalized):
        raise TypeError("normal source capacity requires canonical capacity plans")
    names = tuple(plan.pool_name for plan in normalized)
    if len(set(names)) != len(names):
        raise ValueError("normal source capacity requires unique scheduler pools")
    policy_config.validate_scheduler_pools(names)
    if (not isinstance(pool_ranks, Mapping) or set(pool_ranks) != set(names)
            or any(not isinstance(ranks, tuple) or not ranks
                or any(type(rank) is not int or rank < 0 for rank in ranks)
                for ranks in pool_ranks.values())):
        raise ValueError("normal source capacity requires exact physical rank placement")
    physical_ranks = tuple(rank for ranks in pool_ranks.values() for rank in ranks)
    if len(set(physical_ranks)) != len(physical_ranks):
        raise ValueError("normal source capacity pools cannot share physical ranks")
    reported, total = source_memory["reported_steps"], source_memory["total_simulation_steps"]
    if (type(reported) is not int or type(total) is not int or total <= 0
            or reported != total or source_memory["all_steps_reported"] is not True):
        raise ValueError("normal source capacity requires every executed forward's memory report")
    if source_memory["complete"] is not False or source_memory["normal_capacity_accepted"] is not False:
        raise ValueError("source peaks cannot claim complete normal capacity")
    source_pools = source_memory["pools"]
    if not isinstance(source_pools, Mapping) or set(source_pools) != set(names):
        raise ValueError("normal source capacity must cover every source pool exactly")
    reconciled = {}
    for plan in normalized:
        if plan.source_mode != "analytical" or plan.native_allocation is None:
            raise ValueError("normal source capacity cannot validate debug block-count capacity")
        policy = policy_config.policy_for_pool(plan.pool_name)
        binding = policy.binding
        if (plan.hardware_id, plan.model_config_sha256, plan.tensor_parallel_size,
                plan.pipeline_parallel_size, plan.expert_parallel_size, plan.worker_role,
                plan.kv_block_size_tokens, plan.native_allocation.max_model_len) != (
                binding.hardware_id, binding.model_config_sha256, binding.tensor_parallel_size,
                binding.pipeline_parallel_size, binding.expert_parallel_size, binding.worker_role,
                binding.kv_block_size_tokens, binding.max_model_len):
            raise ValueError("normal source capacity plan disagrees with its runtime policy binding")
        placement = pool_ranks[plan.pool_name]
        records = source_pools[plan.pool_name]["ranks"]
        if (len(placement) != len(plan.rank_budgets) or not isinstance(records, Mapping)
                or set(records) != {str(rank) for rank in placement}):
            raise ValueError("normal source capacity requires every physical rank of its pool")
        activation = next(item.bytes for item in policy.non_kv_reserves if item.kind == "activation")
        other_reserves = policy.non_kv_reserve_bytes - activation
        rank_results = {}
        for budget in plan.rank_budgets:
            physical_rank = placement[budget.rank]
            record = records[str(physical_rank)]
            peak, step_id = _validated_source_peak_witness(record)
            if (budget.non_kv_reserve_bytes != policy.non_kv_reserve_bytes
                    or budget.utilized_hbm_bytes != policy.utilized_hbm_bytes(budget.physical_hbm_bytes)):
                raise ValueError("normal source capacity rank budget lost its bound runtime reserves")
            native_bytes = plan.native_allocation.worker_allocated_tensor_bytes[budget.rank]
            headroom = (budget.utilized_hbm_bytes - budget.model_weight_resident_bytes
                - native_bytes - other_reserves)
            if peak > activation:
                raise ValueError(f"source activation exceeds normal activation budget: pool={plan.pool_name} "
                    f"rank={physical_rank} step={step_id} peak={peak} budget={activation}")
            if peak > headroom:
                raise ValueError(f"source activation exceeds normal non-KV headroom: pool={plan.pool_name} "
                    f"rank={physical_rank} step={step_id} peak={peak} headroom={headroom}")
            if native_bytes > budget.available_kv_bytes:
                raise ValueError("native KV backing exceeds its normal rank budget")
            rank_results[str(physical_rank)] = {
                "capacity_local_rank": budget.rank, "known_activation_peak_bytes": peak,
                "witness_step_id": step_id, "activation_budget_bytes": activation,
                "model_weight_resident_bytes": budget.model_weight_resident_bytes,
                "native_kv_backing_bytes": native_bytes, "other_reserve_bytes": other_reserves,
                "utilized_hbm_bytes": budget.utilized_hbm_bytes,
                "source_non_kv_headroom_bytes": headroom,
                "activation_budget_margin_bytes": activation - peak,
                "headroom_margin_bytes": headroom - peak}
        reconciled[plan.pool_name] = {"ranks": rank_results,
            "policy_evidence_status": policy.evidence_status}
    return {"reported_steps": reported, "total_simulation_steps": total,
        "pools": reconciled, "normal_capacity_accepted": False,
        "scope": "executed_forward_known_source_peaks_fit_existing_normal_budget_only",
        "limitations": ("executed_request_maximum_is_not_a_startup_or_all_workloads_bound",
            "unrepresented_compiler_workspace_and_runtime_memory_remain_unknown",
            "existing_policy_reserves_and_evidence_status_are_unchanged")}


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
                    "evidence_status": policy.evidence_status,
                    **({"runtime_visible_hbm": {"bytes": policy.runtime_visible_hbm_bytes,
                        "provenance": asdict(policy.runtime_visible_hbm_provenance)}}
                        if policy.runtime_visible_hbm_bytes is not None else {}),
                }
            ),
            "rank_budgets": [asdict(budget) for budget in plan.rank_budgets],
            "native_vllm_allocation": (
                None
                if plan.native_allocation is None
                else asdict(plan.native_allocation)
            ),
        })
        if plan.v4_weight_storage_profile is not None:
            entries[-1]["v4_weight_storage_profile"] = plan.v4_weight_storage_profile
            entries[-1]["kv_geometry_semantics"] = "native_shared_pool_row_bytes; bytes_per_token is logical_average_rounded_up"
    return {
        "schema_version": 2,
        "source": {
            "mode": next(iter(modes)),
            "runtime_memory_policy_path": source_path,
            "runtime_memory_policy_sha256": source_sha256,
        },
        "scheduler_pools": entries,
    }
