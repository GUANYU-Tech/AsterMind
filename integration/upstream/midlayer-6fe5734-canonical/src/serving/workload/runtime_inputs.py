"""Run-time workload quantities that are not static model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from .expert_frequency import ExpertFrequencyProfile
from .graph_padding import GraphExecutionConfig


_ROUTING_REPLAY_ROOT_FIELDS = {
    "schema_version",
    "artifact_id",
    "canonical_payload_sha256",
    "provenance",
    "context",
    "records",
}
_ROUTING_REPLAY_PROVENANCE_FIELDS = {
    "producer",
    "vllm_version",
    "vllm_source_commit",
    "vllm_package_manifest_sha256",
    "model_config_sha256",
}
_ROUTING_REPLAY_CONTEXT_FIELDS = {
    "model_config_sha256",
    "request_stream_sha256",
    "scheduler_config_sha256",
    "pool_name",
    "tensor_parallel_size",
    "expert_parallel_size",
    "expert_rank_group",
    "expert_placement_strategy",
    "num_experts",
    "topk",
}
_ROUTING_REPLAY_RECORD_FIELDS = {
    "step_id",
    "layer_index",
    "moe_invocation_id",
    "schedule_step_sha256",
    "tokens",
    "tokens_per_expert",
}
_SEEDED_ROUTING_ALGORITHM = (
    "sha256_weighted_rejection_topk_without_replacement_model_step_v2")
_PROFILE_ROUTING_ALGORITHM = (
    "sha256_profile_weighted_topk_without_replacement_model_step_v1")
_UNIFORM_ROUTING_DISTRIBUTION = "maximum_entropy_uniform_topk_v1"
_SKEW_ROUTING_DISTRIBUTION = "stable_hotset_skew_sensitivity_v1"
_PROFILE_ROUTING_DISTRIBUTION = "model_profile_layer_mean_v1"
# Per-MoE-layer realization of the same imported profile: each MoE layer's
# histogram is drawn from that layer's own row (the router of every layer is
# a separate gate); the MTP layer, which the profile does not cover, draws
# from the model-level mean row.  One histogram is shared by every step with
# the same MoE layer and token count (the price depends only on the
# histogram), so it is drawn once per (layer, tokens) signature.
_PROFILE_LAYER_ROUTING_ALGORITHM = (
    "sha256_seeded_pcg64_gumbel_topk_per_moe_layer_v1")
_PROFILE_LAYER_ROUTING_DISTRIBUTION = "model_profile_per_moe_layer_v1"
_PROFILE_ROUTING_REALIZATIONS = {
    _PROFILE_ROUTING_DISTRIBUTION: (
        _PROFILE_ROUTING_ALGORITHM, "model_step_block_signature"),
    _PROFILE_LAYER_ROUTING_DISTRIBUTION: (
        _PROFILE_LAYER_ROUTING_ALGORITHM, "model_moe_layer_token_count"),
}
_REPLAY_ROUTING_DISTRIBUTION = "captured_expert_histogram_v1"
_REPLAY_ROUTING_ALGORITHM = "expert_histogram_replay_v1"
_ROUTING_PROVENANCE_FIELDS = {
    "schema_version",
    "source_kind",
    "source_id",
    "algorithm",
    "distribution",
    "hot_expert_count",
    "hot_expert_weight",
    "root_seed",
    "policy_descriptor_sha256",
    "replay_source_sha256",
    "replay_canonical_payload_sha256",
    "record_sha256",
    "step_id",
    "layer_index",
    "moe_invocation_id",
    "schedule_step_sha256",
    "mechanism_source",
    "validation_status",
}
_PROFILE_ROUTING_PROVENANCE_FIELDS = _ROUTING_PROVENANCE_FIELDS | {
    "profile_sha256",
    "profile_manifest_sha256",
    "sharing_scope",
    "execution_signature_sha256",
}


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    normalized = {}
    for key, value in pairs:
        if key in normalized:
            raise ValueError(
                f"duplicate expert routing replay JSON field: {key!r}")
        normalized[key] = value
    return normalized


def _require_exact_fields(
    value: object,
    expected: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"expert routing replay {label} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"expert routing replay {label} fields must be exactly "
            f"{sorted(expected)}, got {sorted(value)}")
    return value


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"expert routing replay {name} must be a non-empty string")
    return value


def _require_integer(name: str, value: object, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expert routing replay {name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"expert routing replay {name} must be positive")
    if not positive and value < 0:
        raise ValueError(
            f"expert routing replay {name} cannot be negative")
    return value


def _require_sha256(name: str, value: object) -> str:
    normalized = _require_string(name, value)
    if (len(normalized) != 64
            or any(character not in "0123456789abcdef"
                   for character in normalized)):
        raise ValueError(
            f"expert routing replay {name} must be a lowercase SHA256")
    return normalized


def _validate_expert_histogram(
    tokens_per_expert: Tuple[int, ...],
    *,
    tokens: int,
    num_experts: int,
    topk: int,
    label: str,
) -> Tuple[int, ...]:
    counts = tuple(tokens_per_expert)
    if len(counts) != num_experts:
        raise ValueError(f"{label} count length must match num_experts")
    if any(isinstance(count, bool) or not isinstance(count, int)
           or count < 0 for count in counts):
        raise ValueError(f"{label} counts must be non-negative integers")
    expected_routed_tokens = tokens * topk
    if sum(counts) != expected_routed_tokens:
        raise ValueError(
            f"{label} counts must sum to tokens * top-k: "
            f"{sum(counts)} != {expected_routed_tokens}")
    if any(count > tokens for count in counts):
        raise ValueError(
            f"{label} count cannot exceed tokens for any expert")
    return counts


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExpertRoutingReplayProvenance:
    producer: str
    vllm_version: str
    vllm_source_commit: str
    vllm_package_manifest_sha256: str
    model_config_sha256: str

    def __post_init__(self) -> None:
        if self.producer != "vllm_model_runner_observation":
            raise ValueError(
                "expert routing replay producer must be "
                "'vllm_model_runner_observation'")
        _require_string("vllm_version", self.vllm_version)
        commit = _require_string(
            "vllm_source_commit", self.vllm_source_commit)
        if (len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(
                "expert routing replay vllm_source_commit must be a "
                "lowercase 40-character Git commit")
        _require_sha256(
            "vllm_package_manifest_sha256",
            self.vllm_package_manifest_sha256,
        )
        _require_sha256(
            "model_config_sha256", self.model_config_sha256)

    def manifest(self) -> dict[str, object]:
        return {
            "producer": self.producer,
            "vllm_version": self.vllm_version,
            "vllm_source_commit": self.vllm_source_commit,
            "vllm_package_manifest_sha256": (
                self.vllm_package_manifest_sha256),
            "model_config_sha256": self.model_config_sha256,
        }


@dataclass(frozen=True)
class ExpertRoutingReplayContext:
    model_config_sha256: str
    request_stream_sha256: str
    scheduler_config_sha256: str
    pool_name: str
    tensor_parallel_size: int
    expert_parallel_size: int
    expert_rank_group: Tuple[int, ...]
    expert_placement_strategy: str
    num_experts: int
    topk: int

    def __post_init__(self) -> None:
        _require_sha256("model_config_sha256", self.model_config_sha256)
        _require_sha256(
            "request_stream_sha256", self.request_stream_sha256)
        _require_sha256(
            "scheduler_config_sha256", self.scheduler_config_sha256)
        _require_string("pool_name", self.pool_name)
        for name in (
                "tensor_parallel_size", "expert_parallel_size",
                "num_experts", "topk"):
            _require_integer(name, getattr(self, name), positive=True)
        if not isinstance(self.expert_rank_group, (list, tuple)):
            raise TypeError(
                "expert routing replay expert_rank_group must be a sequence")
        rank_group = tuple(self.expert_rank_group)
        if len(rank_group) != self.expert_parallel_size:
            raise ValueError(
                "expert routing replay expert rank group size disagrees with "
                "expert_parallel_size")
        if any(isinstance(rank, bool) or not isinstance(rank, int)
               for rank in rank_group):
            raise TypeError(
                "expert routing replay expert ranks must be integers")
        if any(rank < 0 for rank in rank_group):
            raise ValueError(
                "expert routing replay expert ranks cannot be negative")
        if len(rank_group) != len(set(rank_group)):
            raise ValueError(
                "expert routing replay expert ranks must be unique")
        object.__setattr__(self, "expert_rank_group", rank_group)
        if self.expert_placement_strategy != "linear":
            raise ValueError(
                "expert routing replay currently requires linear expert "
                "placement")
        if self.topk > self.num_experts:
            raise ValueError(
                "expert routing replay topk cannot exceed num_experts")

    def manifest(self) -> dict[str, object]:
        return {
            "model_config_sha256": self.model_config_sha256,
            "request_stream_sha256": self.request_stream_sha256,
            "scheduler_config_sha256": self.scheduler_config_sha256,
            "pool_name": self.pool_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "expert_rank_group": self.expert_rank_group,
            "expert_placement_strategy": self.expert_placement_strategy,
            "num_experts": self.num_experts,
            "topk": self.topk,
        }


@dataclass(frozen=True)
class ExpertRoutingReplayRecord:
    step_id: str
    layer_index: int
    moe_invocation_id: str
    schedule_step_sha256: str
    tokens: int
    tokens_per_expert: Tuple[int, ...]

    def __post_init__(self) -> None:
        _require_string("record step_id", self.step_id)
        _require_integer("record layer_index", self.layer_index, positive=False)
        _require_string("record moe_invocation_id", self.moe_invocation_id)
        expected_invocation_id = (
            f"{self.step_id}:layer-{self.layer_index}:moe")
        if self.moe_invocation_id != expected_invocation_id:
            raise ValueError(
                "expert routing replay record invocation ID disagrees with "
                "its step/layer identity: "
                f"{self.moe_invocation_id!r} != {expected_invocation_id!r}")
        _require_sha256(
            "record schedule_step_sha256", self.schedule_step_sha256)
        _require_integer("record tokens", self.tokens, positive=True)
        if not isinstance(self.tokens_per_expert, (list, tuple)):
            raise TypeError(
                "expert routing replay tokens_per_expert must be a sequence")
        counts = tuple(self.tokens_per_expert)
        if not counts:
            raise ValueError(
                "expert routing replay tokens_per_expert cannot be empty")
        if any(isinstance(count, bool) or not isinstance(count, int)
               for count in counts):
            raise TypeError(
                "expert routing replay token counts must be integers")
        if any(count < 0 for count in counts):
            raise ValueError(
                "expert routing replay token counts cannot be negative")
        object.__setattr__(self, "tokens_per_expert", counts)

    def manifest(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "layer_index": self.layer_index,
            "moe_invocation_id": self.moe_invocation_id,
            "schedule_step_sha256": self.schedule_step_sha256,
            "tokens": self.tokens,
            "tokens_per_expert": self.tokens_per_expert,
        }

    @property
    def record_sha256(self) -> str:
        return _canonical_json_sha256(self.manifest())


@dataclass(frozen=True, init=False)
class ExpertRoutingReplay:
    source_path: str
    source_sha256: str
    canonical_payload_sha256: str
    artifact_id: str
    provenance: ExpertRoutingReplayProvenance
    context: ExpertRoutingReplayContext
    records: Tuple[ExpertRoutingReplayRecord, ...]

    @classmethod
    def _from_loaded_payload(
        cls,
        *,
        source_path: str,
        source_sha256: str,
        canonical_payload_sha256: str,
        artifact_id: str,
        provenance: ExpertRoutingReplayProvenance,
        context: ExpertRoutingReplayContext,
        records: Tuple[ExpertRoutingReplayRecord, ...],
    ) -> "ExpertRoutingReplay":
        instance = object.__new__(cls)
        object.__setattr__(instance, "source_path", source_path)
        object.__setattr__(instance, "source_sha256", source_sha256)
        object.__setattr__(
            instance,
            "canonical_payload_sha256",
            canonical_payload_sha256,
        )
        object.__setattr__(instance, "artifact_id", artifact_id)
        object.__setattr__(instance, "provenance", provenance)
        object.__setattr__(instance, "context", context)
        object.__setattr__(instance, "records", records)
        instance._validate()
        return instance

    def _validate(self) -> None:
        _require_string("source_path", self.source_path)
        _require_sha256("source_sha256", self.source_sha256)
        _require_sha256(
            "canonical_payload_sha256", self.canonical_payload_sha256)
        _require_string("artifact_id", self.artifact_id)
        if not isinstance(self.provenance, ExpertRoutingReplayProvenance):
            raise TypeError(
                "expert routing replay provenance has the wrong type")
        if not isinstance(self.context, ExpertRoutingReplayContext):
            raise TypeError("expert routing replay context has the wrong type")
        if self.provenance.model_config_sha256 != (
                self.context.model_config_sha256):
            raise ValueError(
                "expert routing replay provenance and context model hashes "
                "disagree")
        if not isinstance(self.records, (list, tuple)):
            raise TypeError("expert routing replay records must be a sequence")
        records = tuple(self.records)
        if not records or any(
                not isinstance(record, ExpertRoutingReplayRecord)
                for record in records):
            raise TypeError("expert routing replay requires routing records")
        keys = tuple(
            (record.step_id, record.layer_index) for record in records)
        if len(keys) != len(set(keys)):
            raise ValueError(
                "expert routing replay contains duplicate step/layer records")
        invocation_ids = tuple(
            record.moe_invocation_id for record in records)
        if len(invocation_ids) != len(set(invocation_ids)):
            raise ValueError(
                "expert routing replay contains duplicate invocation IDs")
        expected_step_prefix = f"{self.context.pool_name}:vllm:"
        for record in records:
            if not record.step_id.startswith(expected_step_prefix):
                raise ValueError(
                    "expert routing replay step ID disagrees with its "
                    f"Scheduler pool: {record.step_id!r}")
            step_suffix = record.step_id[len(expected_step_prefix):]
            if not step_suffix.isdigit():
                raise ValueError(
                    "expert routing replay step ID must end in a non-negative "
                    "Scheduler step index")
            _validate_expert_histogram(
                record.tokens_per_expert,
                tokens=record.tokens,
                num_experts=self.context.num_experts,
                topk=self.context.topk,
                label="expert routing replay",
            )
        object.__setattr__(self, "records", records)
        expected_payload_sha256 = expert_routing_replay_payload_sha256(
            self.artifact_id,
            self.provenance,
            self.context,
            records,
        )
        if self.canonical_payload_sha256 != expected_payload_sha256:
            raise ValueError(
                "expert routing replay canonical payload SHA256 mismatch: "
                f"declared={self.canonical_payload_sha256}, "
                f"actual={expected_payload_sha256}")

    def record_for(
        self,
        step_id: str,
        layer_index: int,
    ) -> ExpertRoutingReplayRecord:
        matches = tuple(
            record for record in self.records
            if (record.step_id == step_id
                and record.layer_index == layer_index)
        )
        if len(matches) != 1:
            raise ValueError(
                "expert routing replay is missing the exact step/layer "
                f"record: step_id={step_id!r}, layer_index={layer_index}")
        return matches[0]

    def assert_consumed(self, invocation_ids: Tuple[str, ...]) -> None:
        if not isinstance(invocation_ids, (list, tuple)):
            raise TypeError(
                "consumed expert routing invocation IDs must be a sequence")
        consumed = tuple(invocation_ids)
        if any(not isinstance(value, str) or not value for value in consumed):
            raise ValueError(
                "consumed expert routing invocation IDs must be non-empty")
        if len(consumed) != len(set(consumed)):
            raise ValueError(
                "consumed expert routing invocation IDs must be unique")
        expected = tuple(record.moe_invocation_id for record in self.records)
        if set(consumed) != set(expected):
            missing = sorted(set(expected) - set(consumed))
            extra = sorted(set(consumed) - set(expected))
            raise RuntimeError(
                "expert routing replay coverage disagrees with the executed "
                f"MoE invocations: missing={missing}, extra={extra}")

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_id": self.artifact_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "canonical_payload_sha256": self.canonical_payload_sha256,
            "provenance": self.provenance.manifest(),
            "context": self.context.manifest(),
            "record_count": len(self.records),
            "record_ids": tuple(
                record.moe_invocation_id for record in self.records),
        }


def expert_routing_replay_payload_sha256(
    artifact_id: str,
    provenance: ExpertRoutingReplayProvenance,
    context: ExpertRoutingReplayContext,
    records: Tuple[ExpertRoutingReplayRecord, ...],
) -> str:
    """Hash normalized replay semantics, excluding the declared digest."""

    _require_string("artifact_id", artifact_id)
    if not isinstance(provenance, ExpertRoutingReplayProvenance):
        raise TypeError(
            "expert routing replay payload requires canonical provenance")
    if not isinstance(context, ExpertRoutingReplayContext):
        raise TypeError("expert routing replay payload requires context")
    if not isinstance(records, (list, tuple)):
        raise TypeError("expert routing replay payload records must be a sequence")
    normalized_records = tuple(records)
    if any(not isinstance(record, ExpertRoutingReplayRecord)
           for record in normalized_records):
        raise TypeError(
            "expert routing replay payload records must be canonical")
    payload = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "provenance": provenance.manifest(),
        "context": context.manifest(),
        "records": tuple(
            record.manifest()
            for record in sorted(
                normalized_records,
                key=lambda record: (record.step_id, record.layer_index),
            )
        ),
    }
    return _canonical_json_sha256(payload)


def load_expert_routing_replay(path: str | Path) -> ExpertRoutingReplay:
    """Read and fully validate one immutable expert-routing artifact."""

    source_path = Path(path)
    payload = source_path.read_bytes()

    def reject_constant(value: str) -> object:
        raise ValueError(
            f"expert routing replay JSON constant is invalid: {value}")

    raw = json.loads(
        payload,
        object_pairs_hook=_strict_json_object,
        parse_constant=reject_constant,
    )
    root = _require_exact_fields(
        raw, _ROUTING_REPLAY_ROOT_FIELDS, "root")
    schema_version = _require_integer(
        "schema_version", root["schema_version"], positive=True)
    if schema_version != 1:
        raise ValueError("expert routing replay schema_version must be 1")
    provenance_raw = _require_exact_fields(
        root["provenance"],
        _ROUTING_REPLAY_PROVENANCE_FIELDS,
        "provenance",
    )
    context_raw = _require_exact_fields(
        root["context"], _ROUTING_REPLAY_CONTEXT_FIELDS, "context")
    records_raw = root["records"]
    if not isinstance(records_raw, list) or not records_raw:
        raise TypeError(
            "expert routing replay records must be a non-empty list")
    records = []
    for record_index, record_raw in enumerate(records_raw):
        record = _require_exact_fields(
            record_raw,
            _ROUTING_REPLAY_RECORD_FIELDS,
            f"records[{record_index}]",
        )
        records.append(ExpertRoutingReplayRecord(**record))
    return ExpertRoutingReplay._from_loaded_payload(
        source_path=str(source_path.resolve()),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        canonical_payload_sha256=root["canonical_payload_sha256"],
        artifact_id=root["artifact_id"],
        provenance=ExpertRoutingReplayProvenance(**provenance_raw),
        context=ExpertRoutingReplayContext(**context_raw),
        records=tuple(records),
    )


@dataclass(frozen=True)
class ExpertRoutingResolution:
    tokens_per_expert: Tuple[int, ...]
    routing_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens_per_expert, (list, tuple)):
            raise TypeError("resolved expert routing counts must be a sequence")
        counts = tuple(self.tokens_per_expert)
        if not counts or any(
                isinstance(count, bool) or not isinstance(count, int)
                for count in counts):
            raise TypeError(
                "resolved expert routing counts must contain integers")
        if any(count < 0 for count in counts):
            raise ValueError(
                "resolved expert routing counts cannot be negative")
        if not isinstance(self.routing_provenance, Mapping):
            raise TypeError("resolved expert routing provenance must be a mapping")
        object.__setattr__(self, "tokens_per_expert", counts)
        object.__setattr__(
            self,
            "routing_provenance",
            MappingProxyType(dict(self.routing_provenance)),
        )


class ExpertRoutingMemo:
    """Cache shared modeled/profile histograms for one simulation run."""

    def __init__(self) -> None:
        self._counts: dict[tuple[object, ...], Tuple[int, ...]] = {}
        self._hit_count = 0
        self._miss_count = 0

    @property
    def entry_count(self) -> int:
        return len(self._counts)

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    def resolve(self, key: tuple[object, ...], build_counts) -> Tuple[int, ...]:
        if not isinstance(key, tuple):
            raise TypeError("expert routing memo key must be a tuple")
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("expert routing memo key must be hashable") from exc
        if key in self._counts:
            self._hit_count += 1
            return self._counts[key]
        counts = tuple(build_counts())
        self._counts[key] = counts
        self._miss_count += 1
        return counts


def validate_expert_routing_provenance(
    provenance: Mapping[str, object],
    *,
    tokens_per_expert: Tuple[int, ...],
    tokens: int,
    num_experts: int,
    topk: int,
    moe_invocation_id: str,
) -> Mapping[str, object]:
    """Validate graph-level routing provenance against its MoE semantics."""

    if not isinstance(provenance, Mapping):
        raise TypeError("FUSED_MOE routing provenance must be a mapping")
    if "source_kind" not in provenance:
        raise ValueError("FUSED_MOE routing provenance lacks source_kind")
    source_kind = provenance["source_kind"]
    is_profile = source_kind == "profile_observation"
    if is_profile:
        expected_fields = _PROFILE_ROUTING_PROVENANCE_FIELDS
        profile_sha_fields = (
            "profile_sha256",
            "profile_manifest_sha256",
            "execution_signature_sha256",
        )
    else:
        expected_fields = _ROUTING_PROVENANCE_FIELDS
        profile_sha_fields = ()
    if set(provenance) != expected_fields:
        raise ValueError(
            "FUSED_MOE routing provenance fields disagree with the canonical "
            "contract")
    if provenance["schema_version"] != 2:
        raise ValueError("FUSED_MOE routing provenance schema_version must be 2")
    for name in (
            "source_kind", "source_id", "algorithm", "record_sha256",
            "step_id", "moe_invocation_id", "schedule_step_sha256",
            "mechanism_source", "validation_status"):
        if not isinstance(provenance[name], str) or not provenance[name]:
            raise ValueError(
                f"FUSED_MOE routing provenance {name} must be non-empty")
    sha_fields = (
            "record_sha256", "schedule_step_sha256",
            "policy_descriptor_sha256", "replay_source_sha256",
            "replay_canonical_payload_sha256") + profile_sha_fields
    for name in sha_fields:
        value = provenance[name]
        if value is not None:
            _require_sha256(f"routing provenance {name}", value)
    if provenance["moe_invocation_id"] != moe_invocation_id:
        raise ValueError(
            "FUSED_MOE routing provenance invocation ID disagrees with node")
    layer_index = provenance["layer_index"]
    if layer_index is not None and (
            isinstance(layer_index, bool) or not isinstance(layer_index, int)
            or layer_index < 0):
        raise ValueError(
            "FUSED_MOE routing layer_index must be non-negative or null")
    step_id = provenance["step_id"]
    if layer_index is None:
        if (not moe_invocation_id.startswith(f"{step_id}:mtp-pass-")
                or not moe_invocation_id.endswith(":moe")):
            raise ValueError(
                "FUSED_MOE MTP routing invocation disagrees with its step")
    else:
        expected_invocation_id = f"{step_id}:layer-{layer_index}:moe"
        if moe_invocation_id != expected_invocation_id:
            raise ValueError(
                "FUSED_MOE routing invocation disagrees with step/layer: "
                f"{moe_invocation_id!r} != {expected_invocation_id!r}")
    if (isinstance(tokens, bool) or not isinstance(tokens, int)
            or tokens <= 0):
        raise ValueError("FUSED_MOE routing tokens must be positive")
    for name, value in (("num_experts", num_experts), ("topk", topk)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"FUSED_MOE routing {name} must be positive")
    counts = _validate_expert_histogram(
        tokens_per_expert,
        tokens=tokens,
        num_experts=num_experts,
        topk=topk,
        label="FUSED_MOE routing",
    )
    if source_kind == "explicit_assumption":
        if provenance["algorithm"] != _SEEDED_ROUTING_ALGORITHM:
            raise ValueError(
                "modeled FUSED_MOE routing uses a non-canonical algorithm")
        root_seed = provenance["root_seed"]
        if (isinstance(root_seed, bool) or not isinstance(root_seed, int)
                or root_seed < 0):
            raise ValueError(
                "modeled FUSED_MOE routing requires a non-negative seed")
        distribution = provenance["distribution"]
        hot_expert_count = provenance["hot_expert_count"]
        hot_expert_weight = provenance["hot_expert_weight"]
        _validate_seeded_distribution(
            distribution,
            hot_expert_count,
            hot_expert_weight,
            num_experts=num_experts,
        )
        expected_policy_sha256 = _canonical_json_sha256({
            "schema_version": 2,
            "source_kind": source_kind,
            "source_id": provenance["source_id"],
            "algorithm": provenance["algorithm"],
            "distribution": distribution,
            "hot_expert_count": hot_expert_count,
            "hot_expert_weight": hot_expert_weight,
            "root_seed": root_seed,
        })
        if (provenance["policy_descriptor_sha256"]
                != expected_policy_sha256
                or provenance["replay_source_sha256"] is not None
                or provenance["replay_canonical_payload_sha256"] is not None
                or provenance["mechanism_source"] != "Modeled"
                or provenance["validation_status"] != "Projected"):
            raise ValueError(
                "modeled FUSED_MOE routing provenance is inconsistent")
        expected_record_sha256 = _canonical_json_sha256({
            "policy_descriptor_sha256": expected_policy_sha256,
            "step_id": step_id,
            "layer_index": layer_index,
            "moe_invocation_id": moe_invocation_id,
            "schedule_step_sha256": provenance["schedule_step_sha256"],
            "tokens": tokens,
            "num_experts": num_experts,
            "topk": topk,
            "tokens_per_expert": counts,
        })
    elif source_kind == "profile_observation":
        if provenance["distribution"] not in _PROFILE_ROUTING_REALIZATIONS:
            raise ValueError(
                "profile FUSED_MOE routing uses an unknown distribution")
        algorithm, sharing_scope = _PROFILE_ROUTING_REALIZATIONS[
            provenance["distribution"]]
        if provenance["algorithm"] != algorithm:
            raise ValueError(
                "profile FUSED_MOE routing uses a non-canonical algorithm")
        if (provenance["hot_expert_count"] is not None
                or provenance["hot_expert_weight"] is not None
                or provenance["root_seed"] is not None
                or provenance["policy_descriptor_sha256"] is None
                or provenance["replay_source_sha256"] is not None
                or provenance["replay_canonical_payload_sha256"] is not None
                or provenance["sharing_scope"] != sharing_scope
                or provenance["mechanism_source"] != "ImportedProfile"
                or provenance["validation_status"]
                != "SourceProvenanceUnavailable"):
            raise ValueError(
                "profile FUSED_MOE routing provenance is inconsistent")
        expected_record_sha256 = _canonical_json_sha256({
            "policy_descriptor_sha256": provenance["policy_descriptor_sha256"],
            "step_id": step_id,
            "layer_index": layer_index,
            "moe_invocation_id": moe_invocation_id,
            "schedule_step_sha256": provenance["schedule_step_sha256"],
            "tokens": tokens,
            "num_experts": num_experts,
            "topk": topk,
            "tokens_per_expert": counts,
        })
    elif source_kind == "replay_observation":
        if provenance["algorithm"] != _REPLAY_ROUTING_ALGORITHM:
            raise ValueError(
                "replayed FUSED_MOE routing uses a non-canonical algorithm")
        if layer_index is None:
            raise ValueError(
                "expert routing replay v1 cannot identify an MTP layer")
        if (provenance["distribution"] != _REPLAY_ROUTING_DISTRIBUTION
                or provenance["hot_expert_count"] is not None
                or provenance["hot_expert_weight"] is not None
                or provenance["root_seed"] is not None
                or provenance["policy_descriptor_sha256"] is not None
                or provenance["replay_source_sha256"] is None
                or provenance["replay_canonical_payload_sha256"] is None
                or provenance["mechanism_source"] != "Captured"
                or provenance["validation_status"] != "Validated"):
            raise ValueError(
                "replayed FUSED_MOE routing provenance is inconsistent")
        expected_record_sha256 = ExpertRoutingReplayRecord(
            step_id=step_id,
            layer_index=layer_index,
            moe_invocation_id=moe_invocation_id,
            schedule_step_sha256=provenance["schedule_step_sha256"],
            tokens=tokens,
            tokens_per_expert=counts,
        ).record_sha256
    else:
        raise ValueError(
            f"unknown FUSED_MOE routing source: {source_kind!r}")
    if provenance["record_sha256"] != expected_record_sha256:
        raise ValueError(
            "FUSED_MOE routing record SHA256 disagrees with node semantics")
    return MappingProxyType(dict(provenance))


@dataclass(frozen=True)
class ExpertRoutingInput:
    source_kind: str
    source_id: str
    algorithm: str
    root_seed: Optional[int]
    request_stream_sha256: Optional[str]
    replay: Optional[ExpertRoutingReplay]
    vllm_runtime_binding_sha256: Optional[str]
    distribution: str
    hot_expert_count: Optional[int]
    hot_expert_weight: Optional[int]
    profile: Optional[ExpertFrequencyProfile] = None

    def __post_init__(self) -> None:
        if self.source_kind not in (
                "not_applicable", "explicit_assumption",
                "profile_observation", "replay_observation"):
            raise ValueError(
                f"unknown expert routing source kind: {self.source_kind!r}")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("expert routing source_id must be non-empty")
        if not isinstance(self.algorithm, str) or not self.algorithm:
            raise ValueError("expert routing algorithm must be non-empty")
        if not isinstance(self.distribution, str) or not self.distribution:
            raise ValueError("expert routing distribution must be non-empty")
        if self.source_kind == "not_applicable":
            if (self.source_id != "not_applicable"
                    or self.algorithm != "not_applicable"
                    or self.root_seed is not None
                    or self.request_stream_sha256 is not None
                    or self.replay is not None
                    or self.vllm_runtime_binding_sha256 is not None
                    or self.distribution != "not_applicable"
                    or self.hot_expert_count is not None
                    or self.hot_expert_weight is not None
                    or self.profile is not None):
                raise ValueError(
                    "not-applicable expert routing cannot carry a source")
        elif self.source_kind == "explicit_assumption":
            if self.algorithm != _SEEDED_ROUTING_ALGORITHM:
                raise ValueError(
                    "explicit expert routing assumption requires the "
                    f"{_SEEDED_ROUTING_ALGORITHM!r} algorithm")
            if (isinstance(self.root_seed, bool)
                    or not isinstance(self.root_seed, int)):
                raise TypeError(
                    "explicit expert routing root_seed must be an integer")
            if self.root_seed < 0:
                raise ValueError(
                    "explicit expert routing root_seed cannot be negative")
            _validate_seeded_distribution(
                self.distribution,
                self.hot_expert_count,
                self.hot_expert_weight,
                num_experts=None,
            )
            if (self.request_stream_sha256 is not None
                    or self.replay is not None
                    or self.vllm_runtime_binding_sha256 is not None
                    or self.profile is not None):
                raise ValueError(
                    "explicit expert routing assumption cannot carry replay "
                    "inputs")
        elif self.source_kind == "profile_observation":
            if self.distribution not in _PROFILE_ROUTING_REALIZATIONS:
                raise ValueError(
                    "profile expert routing uses an unknown distribution")
            if self.algorithm != _PROFILE_ROUTING_REALIZATIONS[
                    self.distribution][0]:
                raise ValueError(
                    "profile expert routing requires the canonical profile "
                    "sampling algorithm")
            if (self.root_seed is not None
                    or self.request_stream_sha256 is not None
                    or self.replay is not None
                    or self.vllm_runtime_binding_sha256 is not None
                    or self.hot_expert_count is not None
                    or self.hot_expert_weight is not None):
                raise ValueError(
                    "profile expert routing cannot carry assumption or replay inputs")
            if not isinstance(self.profile, ExpertFrequencyProfile):
                raise TypeError(
                    "profile_observation requires an ExpertFrequencyProfile")
            if self.source_id != self.profile.canonical_model_id:
                raise ValueError(
                    "profile expert routing source_id must match canonical model ID")
        elif self.source_kind == "replay_observation":
            if self.algorithm != _REPLAY_ROUTING_ALGORITHM:
                raise ValueError(
                    "expert routing replay requires the "
                    f"{_REPLAY_ROUTING_ALGORITHM!r} algorithm")
            if self.root_seed is not None:
                raise ValueError("expert routing replay cannot carry a seed")
            if (self.distribution != _REPLAY_ROUTING_DISTRIBUTION
                    or self.hot_expert_count is not None
                    or self.hot_expert_weight is not None):
                raise ValueError(
                    "expert routing replay requires captured histogram "
                    "distribution without assumption parameters")
            if self.profile is not None:
                raise ValueError("expert routing replay cannot carry a profile")
            if self.request_stream_sha256 is None:
                raise ValueError(
                    "expert routing replay requires request stream identity")
            _require_sha256(
                "request_stream_sha256", self.request_stream_sha256)
            if not isinstance(self.replay, ExpertRoutingReplay):
                raise TypeError(
                    "replay_observation requires an ExpertRoutingReplay")
            if self.source_id != self.replay.artifact_id:
                raise ValueError(
                    "expert routing source_id must match replay artifact_id")
            if (self.request_stream_sha256
                    != self.replay.context.request_stream_sha256):
                raise ValueError(
                    "expert routing replay request stream SHA256 mismatch")
            if self.vllm_runtime_binding_sha256 is not None:
                _require_sha256(
                    "vllm_runtime_binding_sha256",
                    self.vllm_runtime_binding_sha256,
                )
        else:
            raise ValueError(
                f"unknown expert routing source kind: {self.source_kind!r}")

    @classmethod
    def not_applicable(cls) -> "ExpertRoutingInput":
        return cls(
            "not_applicable", "not_applicable", "not_applicable",
            None, None, None, None, "not_applicable", None, None,
        )

    @classmethod
    def seeded_assumption(
        cls,
        source_id: str,
        root_seed: int,
    ) -> "ExpertRoutingInput":
        return cls(
            "explicit_assumption",
            source_id,
            _SEEDED_ROUTING_ALGORITHM,
            root_seed,
            None,
            None,
            None,
            _UNIFORM_ROUTING_DISTRIBUTION,
            None,
            None,
        )

    @classmethod
    def skewed_assumption(
        cls,
        source_id: str,
        root_seed: int,
        *,
        hot_expert_count: int,
        hot_expert_weight: int,
    ) -> "ExpertRoutingInput":
        return cls(
            "explicit_assumption",
            source_id,
            _SEEDED_ROUTING_ALGORITHM,
            root_seed,
            None,
            None,
            None,
            _SKEW_ROUTING_DISTRIBUTION,
            hot_expert_count,
            hot_expert_weight,
        )

    @classmethod
    def profile_observation(
        cls,
        profile: ExpertFrequencyProfile,
        *,
        per_moe_layer: bool = False,
    ) -> "ExpertRoutingInput":
        if not isinstance(profile, ExpertFrequencyProfile):
            raise TypeError(
                "profile expert routing requires an ExpertFrequencyProfile")
        distribution = (
            _PROFILE_LAYER_ROUTING_DISTRIBUTION if per_moe_layer
            else _PROFILE_ROUTING_DISTRIBUTION)
        return cls(
            "profile_observation",
            profile.canonical_model_id,
            _PROFILE_ROUTING_REALIZATIONS[distribution][0],
            None,
            None,
            None,
            None,
            distribution,
            None,
            None,
            profile,
        )

    @classmethod
    def replay_observation(
        cls,
        replay: ExpertRoutingReplay,
        request_stream_sha256: str,
    ) -> "ExpertRoutingInput":
        if not isinstance(replay, ExpertRoutingReplay):
            raise TypeError(
                "expert routing replay input must be ExpertRoutingReplay")
        return cls(
            "replay_observation",
            replay.artifact_id,
            _REPLAY_ROUTING_ALGORITHM,
            None,
            request_stream_sha256,
            replay,
            None,
            _REPLAY_ROUTING_DISTRIBUTION,
            None,
            None,
        )

    def bind_vllm_runtime(
        self,
        *,
        version: str,
        source_commit: str,
        package_python_manifest_sha256: str,
        model_config_sha256: str,
    ) -> "ExpertRoutingInput":
        """Bind a replay to the effective pinned vLLM runtime."""

        if self.source_kind != "replay_observation" or self.replay is None:
            raise ValueError(
                "only an expert routing replay can bind a vLLM runtime")
        actual = {
            "vllm_version": _require_string("vLLM version", version),
            "vllm_source_commit": _require_string(
                "vLLM source commit", source_commit),
            "vllm_package_manifest_sha256": _require_sha256(
                "vLLM package manifest SHA256",
                package_python_manifest_sha256,
            ),
            "model_config_sha256": _require_sha256(
                "vLLM model config SHA256", model_config_sha256),
        }
        expected = self.replay.provenance.manifest()
        expected.pop("producer")
        if actual != expected:
            raise RuntimeError(
                "expert routing replay provenance disagrees with the "
                f"effective vLLM runtime: expected={expected}, "
                f"actual={actual}")
        binding_sha256 = _canonical_json_sha256({
            "schema_version": 1,
            "producer": self.replay.provenance.producer,
            **actual,
        })
        return ExpertRoutingInput(
            self.source_kind,
            self.source_id,
            self.algorithm,
            self.root_seed,
            self.request_stream_sha256,
            self.replay,
            binding_sha256,
            self.distribution,
            self.hot_expert_count,
            self.hot_expert_weight,
            self.profile,
        )

    @functools.cached_property
    def policy_descriptor_sha256(self) -> Optional[str]:
        """Content address of this routing policy, excluding any step input.

        Every field it hashes is frozen on construction, so the address is a
        constant of the object; the routing resolution reads it several times
        per MoE invocation and would otherwise re-serialize it each time.
        """

        if self.source_kind == "profile_observation":
            if self.profile is None:
                raise RuntimeError("profile expert routing lost its profile")
            return _canonical_json_sha256({
                "schema_version": 1,
                "source_kind": self.source_kind,
                "source_id": self.source_id,
                "algorithm": self.algorithm,
                "distribution": self.distribution,
                "profile_sha256": self.profile.sha256,
                "profile_manifest_sha256": self.profile.manifest_sha256,
                "sharing_scope": _PROFILE_ROUTING_REALIZATIONS[
                    self.distribution][1],
            })
        if self.source_kind != "explicit_assumption":
            return None
        return _canonical_json_sha256({
            "schema_version": 2,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "algorithm": self.algorithm,
            "distribution": self.distribution,
            "hot_expert_count": self.hot_expert_count,
            "hot_expert_weight": self.hot_expert_weight,
            "root_seed": self.root_seed,
        })

    def assert_request_stream_sha256(self, actual_sha256: str) -> None:
        _require_sha256("actual request_stream_sha256", actual_sha256)
        if (self.source_kind == "replay_observation"
                and actual_sha256 != self.request_stream_sha256):
            raise ValueError(
                "expert routing replay request stream SHA256 disagrees with "
                "SimulationConfig.requests")

    def assert_scheduler_config(
        self,
        *,
        pool_name: str,
        scheduler_config_sha256: str,
    ) -> None:
        if not isinstance(pool_name, str) or not pool_name:
            raise ValueError("expert routing Scheduler pool must be non-empty")
        _require_sha256(
            "actual scheduler_config_sha256", scheduler_config_sha256)
        if self.source_kind != "replay_observation":
            return
        if self.replay is None:
            raise RuntimeError(
                "expert routing replay source lost its replay payload")
        if pool_name != self.replay.context.pool_name:
            raise ValueError(
                "expert routing replay Scheduler pool disagrees with the "
                "executed backend")
        if scheduler_config_sha256 != (
                self.replay.context.scheduler_config_sha256):
            raise ValueError(
                "expert routing replay Scheduler config SHA256 mismatch")

    def resolve(
        self,
        *,
        step_id: str,
        layer_index: Optional[int],
        moe_invocation_id: str,
        schedule_step_sha256: str,
        tokens: int,
        num_experts: int,
        topk: int,
        model_config_sha256: str,
        pool_name: str,
        tensor_parallel_size: int,
        expert_parallel_size: int,
        expert_rank_group: Tuple[int, ...],
        expert_placement_strategy: str,
        memo: Optional[ExpertRoutingMemo] = None,
        moe_layer_ordinal: Optional[int] = None,
    ) -> ExpertRoutingResolution:
        _require_string("resolution step_id", step_id)
        _require_string("resolution moe_invocation_id", moe_invocation_id)
        _require_sha256(
            "resolution schedule_step_sha256", schedule_step_sha256)
        for name, value in (
                ("tokens", tokens), ("num_experts", num_experts),
                ("topk", topk),
                ("tensor_parallel_size", tensor_parallel_size),
                ("expert_parallel_size", expert_parallel_size)):
            _require_integer(name, value, positive=True)
        if topk > num_experts:
            raise ValueError("expert routing topk cannot exceed num_experts")
        _require_sha256("model_config_sha256", model_config_sha256)
        _require_string("pool_name", pool_name)
        if expert_placement_strategy != "linear":
            raise ValueError(
                "expert routing currently requires linear expert placement")
        if memo is not None and not isinstance(memo, ExpertRoutingMemo):
            raise TypeError("expert routing memo must be ExpertRoutingMemo")
        rank_group = tuple(expert_rank_group)
        if len(rank_group) != expert_parallel_size:
            raise ValueError(
                "expert routing rank group size disagrees with "
                "expert_parallel_size")
        if self.source_kind == "not_applicable":
            raise ValueError(
                "MoE workload requires explicit expert routing input")
        if self.source_kind == "explicit_assumption":
            _validate_seeded_distribution(
                self.distribution,
                self.hot_expert_count,
                self.hot_expert_weight,
                num_experts=num_experts,
            )
            routing_group_id = _shared_routing_group_id(
                step_id=step_id,
                layer_index=layer_index,
                tokens=tokens,
                num_experts=num_experts,
                topk=topk,
                model_config_sha256=model_config_sha256,
                pool_name=pool_name,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
                expert_rank_group=rank_group,
                expert_placement_strategy=expert_placement_strategy,
            )
            def build_counts():
                return seeded_expert_token_counts(
                    tokens,
                    num_experts,
                    topk,
                    source_id=self.source_id,
                    root_seed=self.root_seed,
                    moe_invocation_id=routing_group_id,
                    distribution=self.distribution,
                    hot_expert_count=self.hot_expert_count,
                    hot_expert_weight=self.hot_expert_weight,
                )

            if memo is None:
                counts = build_counts()
            else:
                counts = memo.resolve((
                    self.source_kind,
                    self.policy_descriptor_sha256,
                    routing_group_id,
                ), build_counts)
            record_sha256 = _canonical_json_sha256({
                "policy_descriptor_sha256": self.policy_descriptor_sha256,
                "step_id": step_id,
                "layer_index": layer_index,
                "moe_invocation_id": moe_invocation_id,
                "schedule_step_sha256": schedule_step_sha256,
                "tokens": tokens,
                "num_experts": num_experts,
                "topk": topk,
                "tokens_per_expert": counts,
            })
            provenance = {
                "schema_version": 2,
                "source_kind": self.source_kind,
                "source_id": self.source_id,
                "algorithm": self.algorithm,
                "distribution": self.distribution,
                "hot_expert_count": self.hot_expert_count,
                "hot_expert_weight": self.hot_expert_weight,
                "root_seed": self.root_seed,
                "policy_descriptor_sha256": (
                    self.policy_descriptor_sha256),
                "replay_source_sha256": None,
                "replay_canonical_payload_sha256": None,
                "record_sha256": record_sha256,
                "step_id": step_id,
                "layer_index": layer_index,
                "moe_invocation_id": moe_invocation_id,
                "schedule_step_sha256": schedule_step_sha256,
                "mechanism_source": "Modeled",
                "validation_status": "Projected",
            }
        elif self.source_kind == "profile_observation":
            if self.profile is None:
                raise RuntimeError("profile expert routing lost its profile")
            if len(self.profile.mean_probabilities) != num_experts:
                raise ValueError(
                    "expert-frequency profile width disagrees with num_experts")
            routing_group_id = _shared_routing_group_id(
                step_id=step_id,
                layer_index=layer_index,
                tokens=tokens,
                num_experts=num_experts,
                topk=topk,
                model_config_sha256=model_config_sha256,
                pool_name=pool_name,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
                expert_rank_group=rank_group,
                expert_placement_strategy=expert_placement_strategy,
            )
            per_layer = (
                self.distribution == _PROFILE_LAYER_ROUTING_DISTRIBUTION)
            if per_layer:
                if layer_index is None:
                    if moe_layer_ordinal is not None:
                        raise ValueError(
                            "an MTP MoE invocation has no profile layer row")
                    probabilities = self.profile.mean_probabilities
                    layer_key = "mtp_model_mean"
                else:
                    _require_integer(
                        "moe_layer_ordinal", moe_layer_ordinal,
                        positive=False)
                    if moe_layer_ordinal >= len(
                            self.profile.layer_probabilities):
                        raise ValueError(
                            "moe_layer_ordinal exceeds the profile layers")
                    probabilities = self.profile.layer_probabilities[
                        moe_layer_ordinal]
                    layer_key = moe_layer_ordinal
                # Drop the step from the signature: the histogram is a
                # function of the MoE layer and its token count.
                routing_group_id = _canonical_json_sha256({
                    "schema_version": 1,
                    "moe_layer": layer_key,
                    "tokens": tokens,
                    "num_experts": num_experts,
                    "topk": topk,
                    "model_config_sha256": model_config_sha256,
                    "pool_name": pool_name,
                    "tensor_parallel_size": tensor_parallel_size,
                    "expert_parallel_size": expert_parallel_size,
                    "expert_rank_group": rank_group,
                    "expert_placement_strategy": expert_placement_strategy,
                })
            elif moe_layer_ordinal is not None and not isinstance(
                    moe_layer_ordinal, int):
                raise TypeError("moe_layer_ordinal must be an integer")

            def build_counts():
                if per_layer:
                    return profiled_layer_expert_token_counts(
                        tokens,
                        topk,
                        probabilities=probabilities,
                        profile_sha256=self.profile.sha256,
                        routing_group_id=routing_group_id,
                    )
                return profiled_expert_token_counts(
                    tokens,
                    topk,
                    probabilities=self.profile.mean_probabilities,
                    profile_sha256=self.profile.sha256,
                    routing_group_id=routing_group_id,
                )

            if memo is None:
                counts = build_counts()
            else:
                counts = memo.resolve((
                    self.source_kind,
                    self.policy_descriptor_sha256,
                    routing_group_id,
                ), build_counts)
            record_sha256 = _canonical_json_sha256({
                "policy_descriptor_sha256": self.policy_descriptor_sha256,
                "step_id": step_id,
                "layer_index": layer_index,
                "moe_invocation_id": moe_invocation_id,
                "schedule_step_sha256": schedule_step_sha256,
                "tokens": tokens,
                "num_experts": num_experts,
                "topk": topk,
                "tokens_per_expert": counts,
            })
            provenance = {
                "schema_version": 2,
                "source_kind": self.source_kind,
                "source_id": self.source_id,
                "algorithm": self.algorithm,
                "distribution": self.distribution,
                "hot_expert_count": None,
                "hot_expert_weight": None,
                "root_seed": None,
                "policy_descriptor_sha256": self.policy_descriptor_sha256,
                "replay_source_sha256": None,
                "replay_canonical_payload_sha256": None,
                "profile_sha256": self.profile.sha256,
                "profile_manifest_sha256": self.profile.manifest_sha256,
                "sharing_scope": _PROFILE_ROUTING_REALIZATIONS[
                    self.distribution][1],
                "execution_signature_sha256": routing_group_id,
                "record_sha256": record_sha256,
                "step_id": step_id,
                "layer_index": layer_index,
                "moe_invocation_id": moe_invocation_id,
                "schedule_step_sha256": schedule_step_sha256,
                "mechanism_source": "ImportedProfile",
                "validation_status": "SourceProvenanceUnavailable",
            }
        elif self.source_kind == "replay_observation":
            if self.vllm_runtime_binding_sha256 is None:
                raise RuntimeError(
                    "expert routing replay must bind the effective vLLM "
                    "runtime before workload resolution")
            if layer_index is None:
                raise NotImplementedError(
                    "expert routing replay v1 supports only main decoder "
                    "layer invocations")
            if self.replay is None:
                raise RuntimeError(
                    "expert routing replay source lost its replay payload")
            actual_context = ExpertRoutingReplayContext(
                model_config_sha256=model_config_sha256,
                request_stream_sha256=self.request_stream_sha256,
                scheduler_config_sha256=(
                    self.replay.context.scheduler_config_sha256),
                pool_name=pool_name,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
                expert_rank_group=rank_group,
                expert_placement_strategy=expert_placement_strategy,
                num_experts=num_experts,
                topk=topk,
            )
            if actual_context != self.replay.context:
                raise ValueError(
                    "expert routing replay context disagrees with the "
                    "executed model/deployment")
            record = self.replay.record_for(step_id, layer_index)
            if record.moe_invocation_id != moe_invocation_id:
                raise ValueError(
                    "expert routing replay invocation ID disagrees with the "
                    "executed workload")
            if record.schedule_step_sha256 != schedule_step_sha256:
                raise ValueError(
                    "expert routing replay ScheduleStep SHA256 disagrees "
                    "with the executed Scheduler decision")
            if record.tokens != tokens:
                raise ValueError(
                    "expert routing replay token count disagrees with the "
                    f"executed batch: {record.tokens} != {tokens}")
            counts = record.tokens_per_expert
            provenance = {
                "schema_version": 2,
                "source_kind": self.source_kind,
                "source_id": self.source_id,
                "algorithm": self.algorithm,
                "distribution": _REPLAY_ROUTING_DISTRIBUTION,
                "hot_expert_count": None,
                "hot_expert_weight": None,
                "root_seed": None,
                "policy_descriptor_sha256": None,
                "replay_source_sha256": self.replay.source_sha256,
                "replay_canonical_payload_sha256": (
                    self.replay.canonical_payload_sha256),
                "record_sha256": record.record_sha256,
                "step_id": step_id,
                "layer_index": layer_index,
                "moe_invocation_id": moe_invocation_id,
                "schedule_step_sha256": schedule_step_sha256,
                "mechanism_source": "Captured",
                "validation_status": "Validated",
            }
        else:
            raise ValueError(
                f"unknown expert routing source kind: {self.source_kind!r}")
        counts = _validate_expert_histogram(
            counts,
            tokens=tokens,
            num_experts=num_experts,
            topk=topk,
            label="expert routing",
        )
        validated_provenance = validate_expert_routing_provenance(
            provenance,
            tokens_per_expert=counts,
            tokens=tokens,
            num_experts=num_experts,
            topk=topk,
            moe_invocation_id=moe_invocation_id,
        )
        return ExpertRoutingResolution(counts, validated_provenance)

    def assert_replay_consumed(
        self,
        invocation_ids: Tuple[str, ...],
    ) -> None:
        if self.source_kind == "replay_observation":
            if self.replay is None:
                raise RuntimeError(
                    "expert routing replay source lost its replay payload")
            self.replay.assert_consumed(invocation_ids)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "algorithm": self.algorithm,
            "distribution": self.distribution,
            "hot_expert_count": self.hot_expert_count,
            "hot_expert_weight": self.hot_expert_weight,
            "root_seed": self.root_seed,
            "policy_descriptor_sha256": self.policy_descriptor_sha256,
            "vllm_runtime_binding_sha256": (
                self.vllm_runtime_binding_sha256),
            "replay": (
                None if self.replay is None else self.replay.manifest()),
            "profile": (
                None if self.profile is None else self.profile.manifest()),
        }


@dataclass(frozen=True)
class RuntimeInputs:
    expert_routing: ExpertRoutingInput
    attention_kv_layout: str
    attention_kv_distribution: str
    graph_execution: GraphExecutionConfig = field(
        default_factory=GraphExecutionConfig.disabled)
    # Serving framework whose model-runner dispatch the workload follows
    # (e.g. SGLang's MHA-vs-absorbed MLA prefill choice).
    serving_framework: str = "vllm"

    def __post_init__(self) -> None:
        if self.serving_framework not in ("vllm", "sglang"):
            raise ValueError(
                f"unsupported serving_framework: {self.serving_framework!r}")
        if not isinstance(self.expert_routing, ExpertRoutingInput):
            raise TypeError(
                "expert_routing must be an ExpertRoutingInput")
        if not isinstance(self.graph_execution, GraphExecutionConfig):
            raise TypeError(
                "graph_execution must be a GraphExecutionConfig")
        if not isinstance(self.attention_kv_layout, str):
            raise TypeError("attention_kv_layout must be a string")
        if self.attention_kv_layout not in (
                "contiguous", "paged", "paged_varlen", "chunked_prefill", "mla"):
            raise ValueError(
                "unsupported attention_kv_layout: "
                f"{self.attention_kv_layout!r}")
        if not isinstance(self.attention_kv_distribution, str):
            raise TypeError("attention_kv_distribution must be a string")
        if self.attention_kv_distribution not in ("uniform", "skew"):
            raise ValueError(
                "unsupported attention_kv_distribution: "
                f"{self.attention_kv_distribution!r}")


def _shared_routing_group_id(
    *,
    step_id: str,
    layer_index: Optional[int],
    tokens: int,
    num_experts: int,
    topk: int,
    model_config_sha256: str,
    pool_name: str,
    tensor_parallel_size: int,
    expert_parallel_size: int,
    expert_rank_group: Tuple[int, ...],
    expert_placement_strategy: str,
) -> str:
    if layer_index is None:
        block_class = "mtp"
    else:
        block_class = "main_decoder"
    return _canonical_json_sha256({
        "schema_version": 1,
        "step_id": step_id,
        "block_class": block_class,
        "tokens": tokens,
        "num_experts": num_experts,
        "topk": topk,
        "model_config_sha256": model_config_sha256,
        "pool_name": pool_name,
        "tensor_parallel_size": tensor_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "expert_rank_group": tuple(expert_rank_group),
        "expert_placement_strategy": expert_placement_strategy,
    })


def seeded_expert_token_counts(
    num_tokens: int,
    num_experts: int,
    topk: int,
    *,
    source_id: str,
    root_seed: int,
    moe_invocation_id: str,
    distribution: str = _UNIFORM_ROUTING_DISTRIBUTION,
    hot_expert_count: Optional[int] = None,
    hot_expert_weight: Optional[int] = None,
) -> Tuple[int, ...]:
    """Generate a stable maximum-entropy baseline or explicit skew point."""

    for name, value in (
            ("num_tokens", num_tokens), ("num_experts", num_experts),
            ("topk", topk), ("root_seed", root_seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_tokens < 0 or num_experts < 1 or not 1 <= topk <= num_experts:
        raise ValueError("invalid seeded MoE routing dimensions")
    if root_seed < 0:
        raise ValueError("root_seed cannot be negative")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string")
    if not isinstance(moe_invocation_id, str) or not moe_invocation_id:
        raise ValueError("moe_invocation_id must be a non-empty string")
    _validate_seeded_distribution(
        distribution,
        hot_expert_count,
        hot_expert_weight,
        num_experts=num_experts,
    )
    counts = [0] * num_experts
    sample_space = 1 << 64
    prefix = (
        f"midlayer:{_SEEDED_ROUTING_ALGORITHM}:{source_id}:"
        f"{root_seed}:{moe_invocation_id}:{distribution}:"
        f"{hot_expert_count}:{hot_expert_weight}"
    ).encode("utf-8")
    if distribution == _UNIFORM_ROUTING_DISTRIBUTION:
        hot_experts = ()
        cold_experts = tuple(range(num_experts))
        hot_weight = 1
    elif distribution == _SKEW_ROUTING_DISTRIBUTION:
        ranked_experts = tuple(sorted(
            range(num_experts),
            key=lambda expert_id: hashlib.sha256(
                prefix + b":hot-rank:" + str(expert_id).encode("ascii")
            ).digest(),
        ))
        hot_experts = ranked_experts[:hot_expert_count]
        hot_set = set(hot_experts)
        cold_experts = tuple(
            expert_id for expert_id in range(num_experts)
            if expert_id not in hot_set)
        hot_weight = hot_expert_weight
    else:
        raise AssertionError("validated expert routing distribution vanished")
    for token_index in range(num_tokens):
        available_hot = list(hot_experts)
        available_cold = list(cold_experts)
        counter = 0
        for _ in range(topk):
            total_weight = (
                len(available_hot) * hot_weight + len(available_cold))
            unbiased_limit = sample_space - sample_space % total_weight
            while True:
                digest = hashlib.sha256(
                    prefix
                    + b":"
                    + str(token_index).encode("ascii")
                    + b":"
                    + str(counter).encode("ascii")
                ).digest()
                counter += 1
                value = int.from_bytes(digest[:8], "big")
                if value < unbiased_limit:
                    break
            slot = value % total_weight
            hot_slots = len(available_hot) * hot_weight
            if slot < hot_slots:
                expert_id = available_hot.pop(slot // hot_weight)
            else:
                expert_id = available_cold.pop(slot - hot_slots)
            counts[expert_id] += 1
    return tuple(counts)


def profiled_expert_token_counts(
    num_tokens: int,
    topk: int,
    *,
    probabilities: Tuple[float, ...],
    profile_sha256: str,
    routing_group_id: str,
) -> Tuple[int, ...]:
    """Deterministically sample one shared histogram from profile weights."""

    for name, value in (("num_tokens", num_tokens), ("topk", topk)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_tokens < 0 or topk < 1:
        raise ValueError("invalid profiled MoE routing dimensions")
    _require_sha256("profile_sha256", profile_sha256)
    _require_sha256("routing_group_id", routing_group_id)
    weights = tuple(probabilities)
    if not weights:
        raise ValueError("profiled MoE routing requires expert probabilities")
    if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            for weight in weights):
        raise ValueError(
            "profiled MoE routing probabilities must be finite and non-negative")
    if topk > len(weights):
        raise ValueError("profiled MoE routing topk cannot exceed expert count")
    if sum(weight > 0 for weight in weights) < topk:
        raise ValueError(
            "profiled MoE routing requires at least topk positive probabilities")
    counts = [0] * len(weights)
    prefix = (
        f"midlayer:{_PROFILE_ROUTING_ALGORITHM}:{profile_sha256}:"
        f"{routing_group_id}"
    ).encode("utf-8")
    sample_space = 1 << 64
    for token_index in range(num_tokens):
        available = list(range(len(weights)))
        for route_index in range(topk):
            total_weight = sum(weights[expert_id] for expert_id in available)
            if not math.isfinite(total_weight) or total_weight <= 0:
                raise ValueError(
                    "profiled MoE routing exhausted positive expert weights")
            digest = hashlib.sha256(
                prefix
                + b":"
                + str(token_index).encode("ascii")
                + b":"
                + str(route_index).encode("ascii")
            ).digest()
            target = int.from_bytes(digest[:8], "big") / sample_space * total_weight
            cumulative = 0.0
            selected_index = None
            for available_index, expert_id in enumerate(available):
                cumulative += weights[expert_id]
                if target < cumulative:
                    selected_index = available_index
                    break
            if selected_index is None:
                selected_index = len(available) - 1
            expert_id = available.pop(selected_index)
            counts[expert_id] += 1
    return tuple(counts)


def profiled_layer_expert_token_counts(
    num_tokens: int,
    topk: int,
    *,
    probabilities: Tuple[float, ...],
    profile_sha256: str,
    routing_group_id: str,
) -> Tuple[int, ...]:
    """Deterministically sample one MoE layer's histogram from its profile row.

    Each token draws ``topk`` distinct experts, weighted by the row, without
    replacement; the Gumbel-top-k construction gives the same distribution as
    sequential weighted draws.  The PCG64 stream is seeded from the profile
    and the routing group identity.
    """

    import numpy as np

    for name, value in (("num_tokens", num_tokens), ("topk", topk)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_tokens < 0 or topk < 1:
        raise ValueError("invalid profiled MoE routing dimensions")
    _require_sha256("profile_sha256", profile_sha256)
    _require_sha256("routing_group_id", routing_group_id)
    weights = np.asarray(probabilities, dtype=np.float64)
    if weights.ndim != 1 or not weights.size:
        raise ValueError("profiled MoE routing requires expert probabilities")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(
            "profiled MoE routing probabilities must be finite and non-negative")
    if int((weights > 0).sum()) < topk:
        raise ValueError(
            "profiled MoE routing requires at least topk positive probabilities")
    seed = int.from_bytes(hashlib.sha256(
        f"midlayer:{_PROFILE_LAYER_ROUTING_ALGORITHM}:{profile_sha256}:"
        f"{routing_group_id}".encode("utf-8")).digest()[:16], "big")
    generator = np.random.Generator(np.random.PCG64(seed))
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    keys = log_weights[None, :] + generator.gumbel(
        size=(num_tokens, weights.size))
    selected = np.argpartition(-keys, topk - 1, axis=1)[:, :topk]
    counts = np.bincount(selected.ravel(), minlength=weights.size)
    return tuple(int(count) for count in counts)


def _validate_seeded_distribution(
    distribution: object,
    hot_expert_count: object,
    hot_expert_weight: object,
    *,
    num_experts: Optional[int],
) -> None:
    if distribution == _UNIFORM_ROUTING_DISTRIBUTION:
        if hot_expert_count is not None or hot_expert_weight is not None:
            raise ValueError(
                "maximum-entropy uniform routing cannot carry hot-expert "
                "parameters")
        return
    if distribution != _SKEW_ROUTING_DISTRIBUTION:
        raise ValueError(
            f"unknown explicit expert routing distribution: {distribution!r}")
    for name, value in (
            ("hot_expert_count", hot_expert_count),
            ("hot_expert_weight", hot_expert_weight)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"expert routing {name} must be an integer")
    if hot_expert_count < 1:
        raise ValueError("expert routing hot_expert_count must be positive")
    if hot_expert_weight <= 1:
        raise ValueError(
            "expert routing hot_expert_weight must be greater than one")
    if num_experts is not None and hot_expert_count >= num_experts:
        raise ValueError(
            "skew routing hot_expert_count must be smaller than num_experts")
