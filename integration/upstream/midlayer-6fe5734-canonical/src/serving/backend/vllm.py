"""No-GPU adapter for the real vLLM v1 Scheduler control plane.

The native construction path instantiates ``vllm.v1.core.sched.Scheduler``
directly.  It supplies the KV capacity instead of starting a worker profiler,
and feeds fabricated ``ModelRunnerOutput`` objects back after the virtual
execution time has elapsed.  No model runner, weights, CUDA context, KV tensor,
or sampler is created.

All vLLM imports are lazy so the rest of Midlayer remains usable in an
environment that does not install vLLM.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice, repeat
import math
import os
import time
from typing import Any

from src.serving.deployment.deployment_config import (
    worker_role_operator_group,
)
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.backend.mtp_acceptance_replay import (
    COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT,
    MTPAcceptanceReplay,
    MTPAcceptanceSchedulerContext,
    PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT,
    closed_loop_admission_contract_sha256 as _closed_loop_admission_contract_sha256,
    mtp_acceptance_replay_manifest,
    mtp_acceptance_step_from_backend_step,
    normalize_mtp_acceptance_replay_manifest,
    pd_decode_admission_contract_sha256,
)
from src.serving.backend.mtp_acceptance_assumption import (
    MTPAcceptanceAssumption,
    normalize_mtp_acceptance_assumption_manifest,
)
from src.serving.request_gen.request_stream import (
    RequestSpec,
    RequestStream,
    request_stream_sha256,
)
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.schedule.backend_protocol import (
    BackendConfig,
    scheduler_effective_prefix_hit_tokens,
)
from src.serving.schedule.schedule_step import BackendStep, ScheduleStep, ScheduledRequest
from src.serving.host_timing import (
    EngineCoreHostRoundProfile,
    SchedulerHostProfile,
)
from src.serving.workload.model_config import (
    DEEPSEEK_V3_L32_CONFIG_SHA256,
    is_validated_llama_tp2_model_scope,
    is_validated_qwen3_moe_ep2_model_scope,
)
from src.serving.deployment.validated_scope import (
    TARGET_MODEL_PRODUCTION_PARALLELISM,
    is_validated_target_scheduler_scope,
)


# The native Scheduler supplies decisions only. It has no independently
# measured Forward timing service, so local process elapsed time is excluded
# from virtual time rather than sampled at simulation runtime.
_VIRTUAL_SCHEDULER_CPU_S = 0.0


class _LengthOnlyTokenIds(Sequence[int]):
    """Mutable constant-token sequence for the no-model Scheduler adapter."""

    __slots__ = ("_length", "_token_id")

    def __init__(self, token_id: int, length: int) -> None:
        self._token_id = _vllm_integer(token_id, "length-only token ID")
        self._length = _vllm_integer(length, "length-only token count")
        if self._length < 0:
            raise ValueError("vLLM length-only token count cannot be negative")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item):
        if isinstance(item, slice):
            selected = range(self._length)[item]
            return [self._token_id] * len(selected)
        index = item
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError("length-only token index out of range")
        return self._token_id

    def __iter__(self):
        return repeat(self._token_id, self._length)

    def __contains__(self, item) -> bool:
        return self._length > 0 and item == self._token_id

    def __eq__(self, other) -> bool:
        if isinstance(other, _LengthOnlyTokenIds):
            return (self._token_id, self._length) == (
                other._token_id, other._length)
        if isinstance(other, Sequence):
            return len(other) == self._length and all(
                token_id == self._token_id for token_id in other)
        return False

    def copy(self) -> "_LengthOnlyTokenIds":
        return _LengthOnlyTokenIds(self._token_id, self._length)

    def append(self, token_id: int) -> None:
        if token_id != self._token_id:
            raise ValueError(
                "length-only vLLM request received a non-dummy token")
        self._length += 1

    def extend(self, token_ids: Iterable[int]) -> None:
        appended = tuple(token_ids)
        if any(token_id != self._token_id for token_id in appended):
            raise ValueError(
                "length-only vLLM request received a non-dummy token")
        self._length += len(appended)

    def index(self, token_id: int, start: int = 0, stop: int | None = None) -> int:
        if token_id != self._token_id:
            raise ValueError(f"{token_id!r} is not in length-only token IDs")
        normalized_start, normalized_stop, _ = slice(
            start, stop, 1).indices(self._length)
        if normalized_start >= normalized_stop:
            raise ValueError(f"{token_id!r} is not in length-only token IDs")
        return normalized_start


def _vllm_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"vLLM {field} must be an integer")
    return value


def _vllm_request_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("vLLM request ID must be a non-empty string")
    return value


def _validate_mtp_replay_submitted_requests(
    context: MTPAcceptanceSchedulerContext,
    accepted_requests: RequestStream,
) -> None:
    """Bind a replay contract to requests actually admitted by Scheduler."""

    if not isinstance(context, MTPAcceptanceSchedulerContext):
        raise TypeError(
            "MTP replay submitted-request validation requires a Scheduler "
            "context")
    if not isinstance(accepted_requests, RequestStream):
        raise TypeError(
            "MTP replay submitted-request validation requires a RequestStream")
    if context.adapter_contract == COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT:
        actual_identity = request_stream_sha256(accepted_requests)
        expected_identity = context.request_stream_sha256
        identity_name = "request stream"
    elif context.adapter_contract == PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT:
        actual_identity = pd_decode_admission_contract_sha256(
            accepted_requests)
        expected_identity = context.decode_admission_contract_sha256
        identity_name = "decode admission contract"
    else:
        raise RuntimeError("unknown MTP replay adapter contract")
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"MTP acceptance replay {identity_name} hash disagrees with the "
            "requests actually submitted to the Scheduler: "
            f"expected={expected_identity}, actual={actual_identity}")


def _validate_fixed_count_mtp_submitted_requests(
    expected_request_stream_sha256: str,
    accepted_requests: RequestStream,
) -> None:
    """Bind a hash-scoped fixed-count MTP run to submitted requests."""

    if (not isinstance(expected_request_stream_sha256, str)
            or len(expected_request_stream_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in expected_request_stream_sha256)):
        raise ValueError(
            "fixed-count MTP request stream identity must be a SHA256")
    if not isinstance(accepted_requests, RequestStream):
        raise TypeError(
            "fixed-count MTP request validation requires a RequestStream")
    actual_identity = request_stream_sha256(accepted_requests)
    if actual_identity != expected_request_stream_sha256:
        raise RuntimeError(
            "fixed-count MTP request stream hash disagrees with the requests "
            "actually submitted to the Scheduler: "
            f"expected={expected_request_stream_sha256}, "
            f"actual={actual_identity}")


def _admission_order_diagnostic(accepted_requests: RequestStream) -> str:
    """Summarize the actual Scheduler admission order for hash mismatches."""

    request_ids = tuple(
        request.request_id for request in accepted_requests)
    return (
        f"count={len(request_ids)}, "
        f"ids={request_ids!r}")


def _validate_assumption_mtp_submitted_requests(
    assumption: MTPAcceptanceAssumption,
    decode_admission_contract_sha256: str | None,
    accepted_requests: RequestStream,
    closed_loop_admission_contract_sha256: str | None = None,
) -> None:
    """Bind a maximum-entropy policy to actual Scheduler admissions."""

    if not isinstance(assumption, MTPAcceptanceAssumption):
        raise TypeError(
            "maximum-entropy MTP submitted-request validation requires an "
            "assumption")
    if not isinstance(accepted_requests, RequestStream):
        raise TypeError(
            "maximum-entropy MTP request validation requires RequestStream")
    if (decode_admission_contract_sha256 is not None
            and closed_loop_admission_contract_sha256 is not None):
        raise ValueError(
            "maximum-entropy MTP cannot bind both a decode admission "
            "contract and a closed-loop admission contract")
    if closed_loop_admission_contract_sha256 is not None:
        expected_identity = closed_loop_admission_contract_sha256
        actual_identity = _closed_loop_admission_contract_sha256(
            accepted_requests)
        identity_name = "closed-loop admission contract"
    elif decode_admission_contract_sha256 is None:
        expected_identity = assumption.request_stream_sha256
        actual_identity = request_stream_sha256(accepted_requests)
        identity_name = "request stream"
    else:
        expected_identity = decode_admission_contract_sha256
        actual_identity = pd_decode_admission_contract_sha256(
            accepted_requests)
        identity_name = "decode admission contract"
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"maximum-entropy MTP {identity_name} hash disagrees with the "
            "requests actually submitted to the Scheduler: "
            f"expected={expected_identity}, actual={actual_identity}; "
            f"divergence={_admission_order_diagnostic(accepted_requests)}")


def _validate_standalone_af_mtp_preemption_requests(
    accepted_requests: RequestStream,
) -> None:
    """Reject direct-API request streams outside the validated AF slice."""

    if not isinstance(accepted_requests, RequestStream):
        raise TypeError(
            "standalone AF MTP preemption validation requires RequestStream")
    request_shapes = tuple(
        (
            request.arrival_time_s,
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
        )
        for request in accepted_requests
    )
    if request_shapes != (
            (0.0, 32, 2, 0),
            (0.0, 32, 4, 32)):
        raise NotImplementedError(
            "standalone AF MTP preemption direct API requires exactly two "
            "same-time requests shaped (32->2, prefix=0) and "
            "(32->4, prefix=32) in that order")


@dataclass(frozen=True)
class VLLMSchedulerConfig:
    """Inputs needed to construct vLLM's CPU scheduler without a worker."""

    model: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    dtype: str = "bfloat16"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    async_scheduling: bool = False
    dummy_token_id: int = 42
    mtp_num_speculative_tokens: int = 0
    mtp_acceptance_policy: str = "disabled"
    mtp_fixed_accepted_tokens: int = 0
    mtp_acceptance_source_id: str = "disabled"
    mtp_acceptance_replay: MTPAcceptanceReplay | None = None
    mtp_acceptance_assumption: MTPAcceptanceAssumption | None = None
    request_stream_sha256: str | None = None
    decode_admission_contract_sha256: str | None = None
    closed_loop_admission_contract_sha256: str | None = None
    authorized_h200_pd_mtp_sweep: bool = False
    scheduler_host_profile: SchedulerHostProfile | None = None
    engine_core_host_round_profile: EngineCoreHostRoundProfile | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("vLLM model path must be a non-empty string")
        for name in (
                "tensor_parallel_size", "pipeline_parallel_size",
                "dummy_token_id", "mtp_num_speculative_tokens",
                "mtp_fixed_accepted_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"vLLM {name} must be an integer")
        if min(self.tensor_parallel_size, self.pipeline_parallel_size) < 1:
            raise ValueError("vLLM parallel sizes must be positive")
        if self.dummy_token_id < 0:
            raise ValueError("vLLM dummy_token_id cannot be negative")
        if min(
                self.mtp_num_speculative_tokens,
                self.mtp_fixed_accepted_tokens) < 0:
            raise ValueError("vLLM MTP token counts cannot be negative")
        if self.max_model_len is not None:
            if isinstance(self.max_model_len, bool) or not isinstance(
                    self.max_model_len, int):
                raise TypeError("vLLM max_model_len must be an integer or null")
            if self.max_model_len < 1:
                raise ValueError("vLLM max_model_len must be positive")
        if (self.scheduler_host_profile is not None
                and not isinstance(self.scheduler_host_profile,
                                   SchedulerHostProfile)):
            raise TypeError("vLLM Scheduler Host profile is invalid")
        if (self.engine_core_host_round_profile is not None
                and not isinstance(self.engine_core_host_round_profile,
                                   EngineCoreHostRoundProfile)):
            raise TypeError(
                "vLLM EngineCore host-round profile is invalid")
        if (isinstance(self.gpu_memory_utilization, bool)
                or not isinstance(self.gpu_memory_utilization, (int, float))
                or not math.isfinite(self.gpu_memory_utilization)
                or not 0 < self.gpu_memory_utilization <= 1):
            raise ValueError(
                "vLLM gpu_memory_utilization must be finite and in (0, 1]")
        if not isinstance(self.dtype, str):
            raise TypeError("vLLM dtype must be a string")
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError(
                "vLLM dtype must be 'bfloat16' or 'float16', got "
                f"{self.dtype!r}")
        for name in (
                "trust_remote_code", "enable_chunked_prefill",
                "enable_prefix_caching", "enable_expert_parallel",
                "async_scheduling"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"vLLM {name} must be boolean")
        if not isinstance(self.authorized_h200_pd_mtp_sweep, bool):
            raise TypeError("H200 PD MTP sweep authorization must be boolean")
        if not self.enable_chunked_prefill:
            raise ValueError(
                "the pinned vLLM 0.11 Scheduler contract requires chunked "
                "prefill; enable_chunked_prefill must be true")
        if not isinstance(self.mtp_acceptance_policy, str):
            raise TypeError("vLLM MTP acceptance policy must be a string")
        if not isinstance(self.mtp_acceptance_source_id, str):
            raise TypeError("vLLM MTP acceptance source ID must be a string")
        if (self.mtp_acceptance_replay is not None
                and not isinstance(
                    self.mtp_acceptance_replay, MTPAcceptanceReplay)):
            raise TypeError(
                "vLLM MTP acceptance replay must be MTPAcceptanceReplay or null")
        if (self.mtp_acceptance_assumption is not None
                and not isinstance(
                    self.mtp_acceptance_assumption,
                    MTPAcceptanceAssumption)):
            raise TypeError(
                "vLLM MTP acceptance assumption must be "
                "MTPAcceptanceAssumption or null")
        if (self.request_stream_sha256 is not None
                and (not isinstance(self.request_stream_sha256, str)
                     or len(self.request_stream_sha256) != 64
                     or any(character not in "0123456789abcdef"
                            for character in self.request_stream_sha256))):
            raise ValueError(
                "vLLM request stream SHA256 must be a lowercase SHA256 or null")
        if (self.decode_admission_contract_sha256 is not None
                and (not isinstance(
                    self.decode_admission_contract_sha256, str)
                     or len(self.decode_admission_contract_sha256) != 64
                     or any(character not in "0123456789abcdef"
                            for character
                            in self.decode_admission_contract_sha256))):
            raise ValueError(
                "vLLM decode admission contract SHA256 must be a lowercase "
                "SHA256 or null")
        if (self.closed_loop_admission_contract_sha256 is not None
                and (not isinstance(
                    self.closed_loop_admission_contract_sha256, str)
                     or len(self.closed_loop_admission_contract_sha256) != 64
                     or any(character not in "0123456789abcdef"
                            for character
                            in self.closed_loop_admission_contract_sha256))):
            raise ValueError(
                "vLLM closed-loop admission contract SHA256 must be a "
                "lowercase SHA256 or null")
        if (self.closed_loop_admission_contract_sha256 is not None
                and self.decode_admission_contract_sha256 is not None):
            raise ValueError(
                "vLLM closed-loop admission contract cannot coexist with a "
                "decode admission contract")
        if (self.closed_loop_admission_contract_sha256 is not None
                and self.mtp_acceptance_policy != "maximum_entropy"):
            raise ValueError(
                "vLLM closed-loop admission contract requires the "
                "maximum-entropy MTP acceptance policy")
        if self.mtp_num_speculative_tokens == 0:
            if self.mtp_acceptance_policy != "disabled":
                raise ValueError(
                    "disabled vLLM MTP requires acceptance policy 'disabled'")
            if self.mtp_fixed_accepted_tokens != 0:
                raise ValueError(
                    "disabled vLLM MTP cannot accept speculative tokens")
            if self.mtp_acceptance_source_id != "disabled":
                raise ValueError(
                    "disabled vLLM MTP requires source ID 'disabled'")
            if self.mtp_acceptance_replay is not None:
                raise ValueError(
                    "disabled vLLM MTP cannot carry an acceptance replay")
            if self.mtp_acceptance_assumption is not None:
                raise ValueError(
                    "disabled vLLM MTP cannot carry an acceptance assumption")
            if self.request_stream_sha256 is not None:
                raise ValueError(
                    "disabled vLLM MTP cannot carry a request stream hash")
            if self.decode_admission_contract_sha256 is not None:
                raise ValueError(
                    "disabled vLLM MTP cannot carry a decode admission hash")
        elif self.mtp_acceptance_policy == "fixed_count":
            if (self.mtp_fixed_accepted_tokens
                    > self.mtp_num_speculative_tokens):
                raise ValueError(
                    "fixed MTP accepted tokens cannot exceed proposals")
            if (not self.mtp_acceptance_source_id
                    or self.mtp_acceptance_source_id == "disabled"):
                raise ValueError(
                    "fixed MTP acceptance requires a provenance source ID")
            if self.mtp_acceptance_replay is not None:
                raise ValueError(
                    "fixed-count MTP cannot carry an acceptance replay")
            if self.mtp_acceptance_assumption is not None:
                raise ValueError(
                    "fixed-count MTP cannot carry an acceptance assumption")
            if self.decode_admission_contract_sha256 is not None:
                raise ValueError(
                    "fixed-count MTP cannot carry a decode admission hash")
        elif self.mtp_acceptance_policy == "maximum_entropy":
            if self.mtp_fixed_accepted_tokens != 0:
                raise ValueError(
                    "maximum-entropy MTP cannot carry fixed accepted tokens")
            if self.mtp_acceptance_replay is not None:
                raise ValueError(
                    "maximum-entropy MTP cannot carry an acceptance replay")
            if self.mtp_acceptance_assumption is None:
                raise ValueError(
                    "maximum-entropy MTP requires an acceptance assumption")
            assumption = self.mtp_acceptance_assumption
            if self.mtp_acceptance_source_id != assumption.source_id:
                raise ValueError(
                    "maximum-entropy MTP source ID disagrees with the "
                    "assumption")
            if self.mtp_num_speculative_tokens != assumption.proposal_depth:
                raise ValueError(
                    "maximum-entropy MTP proposal depth disagrees with the "
                    "assumption")
            if self.request_stream_sha256 != assumption.request_stream_sha256:
                raise ValueError(
                    "maximum-entropy MTP request stream hash disagrees with "
                    "the assumption")
        elif self.mtp_acceptance_policy == "replay":
            if self.mtp_fixed_accepted_tokens != 0:
                raise ValueError(
                    "replayed MTP cannot carry fixed accepted tokens")
            if self.mtp_acceptance_replay is None:
                raise ValueError(
                    "replayed MTP requires an acceptance replay artifact")
            if self.mtp_acceptance_assumption is not None:
                raise ValueError(
                    "replayed MTP cannot carry an acceptance assumption")
            if self.request_stream_sha256 is None:
                raise ValueError(
                    "replayed MTP requires the actual request stream SHA256")
            replay = self.mtp_acceptance_replay
            if self.mtp_acceptance_source_id != replay.artifact_id:
                raise ValueError(
                    "replayed MTP source ID must equal the artifact ID")
            if (self.mtp_num_speculative_tokens
                    != replay.scheduler_context.num_speculative_tokens):
                raise ValueError(
                    "replayed MTP proposal depth disagrees with the artifact")
            if (self.request_stream_sha256
                    != replay.scheduler_context.ingress_identity_sha256):
                raise ValueError(
                    "replayed MTP request stream hash disagrees with the "
                    "artifact")
            if replay.schema_version == 1:
                if self.decode_admission_contract_sha256 is not None:
                    raise ValueError(
                        "colocated MTP replay cannot carry a decode admission "
                        "hash")
            elif replay.schema_version == 2:
                if (self.decode_admission_contract_sha256
                        != replay.scheduler_context.
                        decode_admission_contract_sha256):
                    raise ValueError(
                        "replayed MTP decode admission hash disagrees with "
                        "the artifact")
            else:
                raise AssertionError("unknown MTP replay schema version")
        else:
            raise ValueError(
                "enabled vLLM MTP requires fixed_count, maximum_entropy, or "
                "replay acceptance")


@dataclass(frozen=True)
class _SchedulerModelView:
    """The text-model fields consumed by the pinned vLLM Scheduler.

    This is deliberately not a second model schema.  Values come from the
    canonical Serving ``ModelConfig`` and expose only the control-plane view
    used by vLLM 0.11.  In particular, constructing this view does not invoke
    vLLM's model registry, tokenizer, model loader, or accelerator platform.
    """

    max_model_len: int
    vocab_size: int
    dtype: Any
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    skip_tokenizer_init: bool = True

    def get_vocab_size(self) -> int:
        return self.vocab_size


@dataclass(frozen=True)
class _SchedulerDraftHFConfig:
    model_type: str
    num_nextn_predict_layers: int


@dataclass(frozen=True)
class _SchedulerDraftModelConfig:
    hf_config: _SchedulerDraftHFConfig
    max_model_len: int


@dataclass(frozen=True)
class _SchedulerSpeculativeConfig:
    method: str
    num_speculative_tokens: int
    draft_model_config: _SchedulerDraftModelConfig
    disable_padded_drafter_batch: bool = False

    def use_eagle(self) -> bool:
        return self.method == "mtp"


@dataclass(frozen=True)
class _SchedulerOnlyVllmConfig:
    """Pinned vLLM Scheduler dependency bundle without ``EngineArgs``."""

    model_config: _SchedulerModelView
    cache_config: Any
    parallel_config: Any
    scheduler_config: Any
    structured_outputs_config: Any
    kv_geometry_representative_pipeline_stage_rank: int
    kv_geometry_representative_layer_start: int
    kv_geometry_representative_layer_end: int
    lora_config: None = None
    speculative_config: _SchedulerSpeculativeConfig | None = None
    kv_transfer_config: None = None
    kv_events_config: None = None


def mtp_scenario_manifest(config: VLLMSchedulerConfig) -> dict[str, object]:
    """Return the canonical, replayable MTP scenario declaration."""

    if not isinstance(config, VLLMSchedulerConfig):
        raise TypeError("MTP scenario requires VLLMSchedulerConfig")
    if config.mtp_acceptance_policy == "disabled":
        validation_status = "disabled"
    elif config.mtp_acceptance_policy in ("fixed_count", "maximum_entropy"):
        validation_status = "projected_explicit_assumption"
    elif config.mtp_acceptance_policy == "replay":
        validation_status = "replayed_external_observation"
    else:
        raise AssertionError(
            "validated MTP config has an unknown acceptance policy: "
            f"{config.mtp_acceptance_policy!r}")
    manifest = {
        "num_speculative_tokens": config.mtp_num_speculative_tokens,
        "acceptance_policy": config.mtp_acceptance_policy,
        "fixed_accepted_tokens": config.mtp_fixed_accepted_tokens,
        "acceptance_source_id": config.mtp_acceptance_source_id,
        "acceptance_replay": (
            None
            if config.mtp_acceptance_replay is None
            else mtp_acceptance_replay_manifest(
                config.mtp_acceptance_replay)
        ),
        "validation_status": validation_status,
    }
    if config.mtp_acceptance_policy == "maximum_entropy":
        if config.mtp_acceptance_assumption is None:
            raise AssertionError(
                "validated maximum-entropy MTP assumption disappeared")
        manifest["acceptance_assumption"] = (
            config.mtp_acceptance_assumption.manifest())
    return manifest


def normalize_mtp_scenario_manifest(value: Any) -> dict[str, object]:
    """Validate an external MTP scenario at its owner boundary."""

    if not isinstance(value, Mapping):
        raise TypeError("MTP scenario must be a mapping")
    raw = dict(value)
    common_fields = {
        "num_speculative_tokens",
        "acceptance_policy",
        "fixed_accepted_tokens",
        "acceptance_source_id",
        "acceptance_replay",
        "validation_status",
    }
    if "acceptance_policy" not in raw:
        raise ValueError("MTP scenario is missing acceptance_policy")
    policy = raw["acceptance_policy"]
    fields = set(common_fields)
    if policy == "maximum_entropy":
        fields.add("acceptance_assumption")
    if set(raw) != fields:
        raise ValueError(
            "MTP scenario fields must be exactly "
            f"{sorted(fields)}, got {sorted(raw)}")
    if policy == "replay":
        replay_manifest = normalize_mtp_acceptance_replay_manifest(
            raw["acceptance_replay"])
        normalized = {
            "num_speculative_tokens": raw["num_speculative_tokens"],
            "acceptance_policy": policy,
            "fixed_accepted_tokens": raw["fixed_accepted_tokens"],
            "acceptance_source_id": raw["acceptance_source_id"],
            "acceptance_replay": replay_manifest,
            "validation_status": "replayed_external_observation",
        }
        if (isinstance(raw["num_speculative_tokens"], bool)
                or not isinstance(raw["num_speculative_tokens"], int)
                or raw["num_speculative_tokens"] <= 0):
            raise ValueError(
                "replayed MTP num_speculative_tokens must be positive")
        if (isinstance(raw["fixed_accepted_tokens"], bool)
                or not isinstance(raw["fixed_accepted_tokens"], int)
                or raw["fixed_accepted_tokens"] != 0):
            raise ValueError(
                "replayed MTP fixed_accepted_tokens must be integer zero")
        if raw["acceptance_source_id"] != replay_manifest["artifact_id"]:
            raise ValueError(
                "replayed MTP source ID must equal the artifact ID")
        if (raw["num_speculative_tokens"]
                != replay_manifest["num_speculative_tokens"]):
            raise ValueError(
                "replayed MTP depth disagrees with the replay manifest")
    elif policy in ("disabled", "fixed_count"):
        if raw["acceptance_replay"] is not None:
            raise ValueError(
                "disabled or fixed-count MTP cannot carry acceptance replay")
        config = VLLMSchedulerConfig(
            "mtp-scenario-contract",
            mtp_num_speculative_tokens=raw["num_speculative_tokens"],
            mtp_acceptance_policy=policy,
            mtp_fixed_accepted_tokens=raw["fixed_accepted_tokens"],
            mtp_acceptance_source_id=raw["acceptance_source_id"],
        )
        normalized = mtp_scenario_manifest(config)
    elif policy == "maximum_entropy":
        if raw["acceptance_replay"] is not None:
            raise ValueError(
                "maximum-entropy MTP cannot carry acceptance replay")
        assumption_manifest = (
            normalize_mtp_acceptance_assumption_manifest(
                raw["acceptance_assumption"]))
        assumption = MTPAcceptanceAssumption(
            assumption_manifest["source_id"],
            assumption_manifest["proposal_depth"],
            assumption_manifest["mean_acceptance_rate"],
            assumption_manifest["root_seed"],
            assumption_manifest["request_stream_sha256"],
        )
        config = VLLMSchedulerConfig(
            "mtp-scenario-contract",
            mtp_num_speculative_tokens=raw["num_speculative_tokens"],
            mtp_acceptance_policy=policy,
            mtp_fixed_accepted_tokens=raw["fixed_accepted_tokens"],
            mtp_acceptance_source_id=raw["acceptance_source_id"],
            mtp_acceptance_assumption=assumption,
            request_stream_sha256=assumption.request_stream_sha256,
        )
        normalized = mtp_scenario_manifest(config)
    else:
        raise ValueError(f"unknown MTP acceptance policy: {policy!r}")
    comparable = dict(raw)
    if policy == "maximum_entropy":
        comparable["acceptance_assumption"] = assumption_manifest
    if comparable != normalized:
        raise ValueError(
            "MTP scenario validation status disagrees with its policy: "
            f"expected={normalized['validation_status']!r}, "
            f"actual={raw['validation_status']!r}")
    return normalized


class VLLMBackend:
    """Framework-independent facade over a vLLM scheduler control plane.

    ``from_model`` is the production constructor.  Direct construction remains
    useful for a version-specific bridge supplied by an embedding application;
    that bridge must already implement the normalized control-plane methods.
    """

    def __init__(self, config: BackendConfig, control_plane: Any) -> None:
        self.config = config
        self._control_plane = control_plane
        self._accepted_request_ids: set[str] = set()

    @classmethod
    def from_model(
        cls,
        config: BackendConfig,
        scheduler_config: VLLMSchedulerConfig,
    ) -> "VLLMBackend":
        _validate_native_scheduler_scope(config, scheduler_config)
        from src.serving.backend.vllm_scheduler_process import (
            VLLMSchedulerProcess,
        )

        control_plane = VLLMSchedulerProcess(config, scheduler_config)
        return cls(config, control_plane)

    @property
    def runtime_info(self) -> Any:
        if not hasattr(self._control_plane, "runtime_info"):
            raise RuntimeError(
                "vLLM runtime provenance is available only for from_model")
        return self._control_plane.runtime_info

    def add_request(self, request: RequestSpec) -> None:
        if request.request_id in self._accepted_request_ids:
            raise ValueError(
                f"duplicate vLLM request ID {request.request_id!r}")
        self._control_plane.add_request(request)
        self._accepted_request_ids.add(request.request_id)

    def add_requests(self, requests: tuple[RequestSpec, ...]) -> None:
        if not isinstance(requests, tuple) or any(
                not isinstance(request, RequestSpec) for request in requests):
            raise TypeError(
                "vLLM bulk admission requires a tuple of RequestSpec objects")
        request_ids = tuple(request.request_id for request in requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("duplicate vLLM request ID in admission batch")
        duplicate_ids = self._accepted_request_ids.intersection(request_ids)
        if duplicate_ids:
            raise ValueError(
                f"duplicate vLLM request IDs: {sorted(duplicate_ids)!r}")
        self._control_plane.add_requests(requests)
        self._accepted_request_ids.update(request_ids)

    def plan_step(self, virtual_time_s: float):
        return self._control_plane.plan_step(virtual_time_s)

    def drain_steps(
        self,
        max_steps: int,
    ) -> tuple[tuple[BackendStep, ...], bool]:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("vLLM tail batch size must be an integer")
        if max_steps < 1:
            raise ValueError("vLLM tail batch size must be positive")
        if hasattr(self._control_plane, "drain_steps"):
            result = self._control_plane.drain_steps(max_steps)
            if (not isinstance(result, tuple) or len(result) != 2
                    or not isinstance(result[0], tuple)
                    or any(not isinstance(step, BackendStep)
                           for step in result[0])
                    or not isinstance(result[1], bool)):
                raise TypeError(
                    "vLLM control plane returned an invalid tail batch")
            if not result[0] and result[1]:
                raise RuntimeError(
                    "vLLM tail batch made no progress while work remains")
            return result

        steps = []
        while len(steps) < max_steps and self.has_work():
            step = self.plan_step(0.0)
            if step is None:
                raise RuntimeError(
                    "vLLM tail has work but plan_step returned no step")
            self.complete_step(step, 0.0)
            steps.append(step)
        return tuple(steps), self.has_work()

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> float:
        return self._control_plane.complete_step(
            step,
            virtual_time_s,
            execute_model=False,
            execute_transfers=False,
            sample_tokens=False,
        )

    def measure_scheduler_service(
        self,
        *,
        sample_count: int,
        requests_per_batch: int,
        prompt_tokens: int,
        output_tokens: int,
    ) -> dict[str, tuple[int, ...]]:
        """Measure native Scheduler calls in its isolated CPU process.

        This is an independent control-plane microbenchmark hook.  It is not
        used by Forward: returned wall-clock samples are for an external
        profile runner to reduce and version before any service value can
        enter the simulation contract.
        """

        if not hasattr(self._control_plane, "measure_scheduler_service"):
            raise RuntimeError(
                "Scheduler service measurement requires the native vLLM "
                "control-plane process")
        result = self._control_plane.measure_scheduler_service(
            sample_count=sample_count,
            requests_per_batch=requests_per_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        if (not isinstance(result, dict)
                or set(result) != {"schedule_ns", "update_ns"}):
            raise TypeError("vLLM Scheduler service measurement is malformed")
        for name, samples in result.items():
            if (not isinstance(samples, tuple)
                    or len(samples) != sample_count
                    or any(isinstance(value, bool)
                           or not isinstance(value, int)
                           or value <= 0 for value in samples)):
                raise TypeError(
                    f"vLLM Scheduler service {name} samples are invalid")
        return result

    def has_work(self) -> bool:
        result = self._control_plane.has_work()
        if not isinstance(result, bool):
            raise TypeError("vLLM control_plane.has_work() must return boolean")
        return result

    def close(self) -> None:
        if not hasattr(self._control_plane, "close"):
            return
        self._control_plane.close()


class _NativeVLLMControlPlane:
    def __init__(self, config: BackendConfig, scheduler_config: VLLMSchedulerConfig) -> None:
        self.config = config
        self.scheduler_config = scheduler_config
        self.scheduler_host_profile = scheduler_config.scheduler_host_profile
        self.engine_core_host_round_profile = (
            scheduler_config.engine_core_host_round_profile)
        self._scheduler_host_uncovered_reported: set[str] = set()
        self.scheduler, self.vllm_config = _build_native_scheduler(
            config, scheduler_config)
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MLAAttentionSpec,
        )
        kv_groups = self.scheduler.kv_cache_config.kv_cache_groups
        if len(kv_groups) != 1:
            raise NotImplementedError(
                "the normalized vLLM bridge requires one KV cache group")
        kv_spec = kv_groups[0].kv_cache_spec
        if isinstance(kv_spec, MLAAttentionSpec):
            self.attention_kv_layout = "mla"
        elif type(kv_spec) is FullAttentionSpec:
            self.attention_kv_layout = "paged_varlen"
        else:
            raise NotImplementedError(
                "the normalized vLLM bridge does not support KV spec "
                f"{type(kv_spec).__name__}")
        self._requests: dict[str, RequestSpec] = {}
        self._deferred_requests: deque[RequestSpec] = deque()
        self._first_schedule: set[str] = set()
        # EngineCore queues scheduled futures in FIFO completion order.  The
        # simulator mirrors that control-plane queue so a pipeline-depth
        # scheduler can submit a later step before output processing releases
        # the earliest slot.
        self._pending: deque[tuple[BackendStep, Any]] = deque()
        self._step_index = 0
        self._accepted_request_ids: set[str] = set()
        self._computed_high_watermark: dict[str, int] = {}
        self._trace_prefix_hit_tokens: dict[str, int] = {}
        self._scheduler_prefix_hit_tokens: dict[str, int] = {}
        self._prefix_replay_active: dict[str, bool] = {}
        self._accepted_requests: list[RequestSpec] = []
        self._mtp_request_scope_validated = False
        self._service_measurement_samples: dict[str, list[int]] | None = None

    def _scheduler_host_service_s(self, schedule, operation: str) -> float:
        profile = self.scheduler_host_profile
        if profile is None:
            return _VIRTUAL_SCHEDULER_CPU_S
        return profile.service_time_s(schedule, operation)

    def _engine_core_host_round_s(self, schedule) -> tuple[float, float]:
        """Return one step's EngineCore submit/wake Host service.

        Hardware without an independently measured EngineCore host-round
        profile is explicitly schema-gated off: the wiring registry only
        registers measured hardware, so those steps carry zero EngineCore
        Host pacing and stay bit-identical.
        """

        profile = self.engine_core_host_round_profile
        if profile is None:
            return 0.0, 0.0
        return profile.service_times_s(schedule)

    def add_request(self, request: RequestSpec) -> None:
        if self._mtp_request_scope_validated:
            raise RuntimeError(
                "hash-bound MTP request scope was already "
                "validated; later request admission is unsupported")
        if request.request_id in self._accepted_request_ids:
            raise ValueError(f"duplicate vLLM request ID {request.request_id!r}")
        self._accepted_requests.append(request)
        self._accepted_request_ids.add(request.request_id)
        self._deferred_requests.append(request)
        self._materialize_waiting_prefix()

    def _materialize_waiting_prefix(self) -> None:
        while (self._deferred_requests
               and len(self.scheduler.waiting) < self.config.max_requests):
            self._materialize_request(self._deferred_requests.popleft())

    def _materialize_request(self, request: RequestSpec) -> None:
        if self.config.is_prefill_only:
            logical_output_limit = min(1, request.output_tokens)
        else:
            logical_output_limit = request.output_tokens
        # Pinned vLLM requires max_tokens > 0 and its prompt-completion step
        # always produces one raw terminal sample.  A length-only Trace may
        # legitimately request zero emitted tokens, so drive that native step
        # with one physical slot and truncate the raw sample at the adapter
        # boundary.  The prompt/KV work remains scheduled and observable.
        native_output_limit = max(1, logical_output_limit)
        native = _make_native_request(
            self.vllm_config,
            request,
            dummy_token_id=self.scheduler_config.dummy_token_id,
            max_output_tokens=native_output_limit,
        )
        _verify_content_hash_state_disabled(native)
        if self.config.is_decode_only:
            if request.prefix_hit_tokens != 0:
                raise NotImplementedError(
                    "decode-only vLLM pools cannot re-inject a trace prefix hit")
            # Metadata-only PD connector: the prefill pool has already produced
            # the first token and the transfer edge has delivered all prompt KV.
            # Seeding these two pieces of native Request state makes vLLM
            # allocate/account the transferred blocks and schedule decode only.
            native.num_computed_tokens = request.input_tokens
            native.append_output_token_ids(self.scheduler_config.dummy_token_id)
            trace_prefix_hit_tokens = 0
            scheduler_prefix_hit_tokens = 0
            computed_high_watermark = native.num_computed_tokens
        else:
            trace_prefix_hit_tokens = request.prefix_hit_tokens
            scheduler_prefix_hit_tokens = scheduler_effective_prefix_hit_tokens(
                request,
                self.config.kv_block_size_tokens,
            )
            # A positive computed count bypasses vLLM's content-hash lookup.
            # The native Scheduler still allocates private blocks atomically
            # with the remaining prompt work, so capacity and admission remain
            # under Scheduler control without fabricating token identities.
            native.num_computed_tokens = scheduler_prefix_hit_tokens
            computed_high_watermark = scheduler_prefix_hit_tokens
        self.scheduler.add_request(native)
        self._requests[request.request_id] = request
        self._computed_high_watermark[request.request_id] = _vllm_integer(
            computed_high_watermark, "initial computed token high-water mark")
        self._trace_prefix_hit_tokens[
            request.request_id] = trace_prefix_hit_tokens
        self._scheduler_prefix_hit_tokens[
            request.request_id] = scheduler_prefix_hit_tokens
        self._prefix_replay_active[request.request_id] = (
            trace_prefix_hit_tokens > scheduler_prefix_hit_tokens)

    def add_requests(self, requests: tuple[RequestSpec, ...]) -> None:
        for request in requests:
            self.add_request(request)

    def drain_steps(
        self,
        max_steps: int,
    ) -> tuple[tuple[BackendStep, ...], bool]:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("vLLM tail batch size must be an integer")
        if max_steps < 1:
            raise ValueError("vLLM tail batch size must be positive")
        steps = []
        while len(steps) < max_steps and self.has_work():
            step = self.plan_step(0.0)
            if step is None:
                raise RuntimeError(
                    "vLLM tail has work but plan_step returned no step")
            self.complete_step(
                step,
                0.0,
                execute_model=False,
                execute_transfers=False,
                sample_tokens=False,
            )
            steps.append(step)
        return tuple(steps), self.has_work()

    def plan_step(self, virtual_time_s: float) -> BackendStep | None:
        if (not self._mtp_request_scope_validated
                and self.scheduler_config.mtp_acceptance_policy == "fixed_count"
                and self.scheduler_config.request_stream_sha256 is not None):
            accepted_requests = RequestStream(self._accepted_requests)
            if (self.config.role == "attention"
                    and self.config.max_requests == 2):
                _validate_standalone_af_mtp_preemption_requests(
                    accepted_requests)
            _validate_fixed_count_mtp_submitted_requests(
                self.scheduler_config.request_stream_sha256,
                accepted_requests,
            )
            self._mtp_request_scope_validated = True
        self._materialize_waiting_prefix()
        if not self.has_work():
            return None

        decision_scope = tuple(self.scheduler.running) + tuple(islice(
            self.scheduler.waiting,
            self.config.max_requests,
        ))
        decision_scope_ids = tuple(
            _vllm_request_id(native_request.request_id)
            for native_request in decision_scope
        )
        if len(decision_scope_ids) != len(set(decision_scope_ids)):
            raise RuntimeError(
                "vLLM decision scope contains duplicate request IDs")
        status_before_native = {
            request_id: native_request.status.name
            for request_id, native_request in zip(
                decision_scope_ids, decision_scope, strict=True)
        }
        computed_before_native = {
            request_id: _vllm_integer(
                native_request.num_computed_tokens,
                "computed token count before schedule",
            )
            for request_id, native_request in zip(
                decision_scope_ids, decision_scope, strict=True)
        }
        blocks_before_native = {
            request_id: self._block_ids(request_id)
            for request_id in decision_scope_ids
        }
        draft_tokens_before_native = {
            request_id: tuple(
                _vllm_integer(token_id, "MTP proposed token ID")
                for token_id in native_request.spec_token_ids
            )
            for request_id, native_request in zip(
                decision_scope_ids, decision_scope, strict=True)
        }
        output_placeholders_before_native = {
            request_id: _vllm_integer(
                native_request.num_output_placeholders,
                "output placeholder count before schedule",
            )
            for request_id, native_request in zip(
                decision_scope_ids, decision_scope, strict=True)
        }
        plan_started_ns = (
            time.perf_counter_ns()
            if self._service_measurement_samples is not None else None)
        native_output = self.scheduler.schedule()
        if plan_started_ns is not None:
            self._service_measurement_samples["schedule_ns"].append(
                time.perf_counter_ns() - plan_started_ns)
        status_after_native = {}
        computed_after_native = {}
        blocks_after_native = {}
        preempted_request_ids = []
        for request_id, computed_before in computed_before_native.items():
            if request_id not in self.scheduler.requests:
                raise RuntimeError(
                    "vLLM removed a request during schedule() before model "
                    f"output: {request_id!r}")
            native_request = self.scheduler.requests[request_id]
            computed_after = _vllm_integer(
                native_request.num_computed_tokens,
                "computed token count after schedule",
            )
            status_name = native_request.status.name
            status_after_native[request_id] = status_name
            computed_after_native[request_id] = computed_after
            blocks_after_native[request_id] = self._block_ids(request_id)
            newly_preempted = (
                status_before_native[request_id] != "PREEMPTED"
                and status_name == "PREEMPTED"
            )
            if computed_after < computed_before:
                if status_name != "PREEMPTED" or computed_after != 0:
                    raise RuntimeError(
                        "vLLM regressed computed tokens without canonical "
                        f"recompute preemption for {request_id!r}: "
                        f"before={computed_before}, after={computed_after}, "
                        f"status_before={status_before_native[request_id]!r}, "
                        f"status_after={status_name!r}, "
                        f"blocks_before={blocks_before_native[request_id]}, "
                        f"blocks_after={blocks_after_native[request_id]}")
                newly_preempted = True
            if newly_preempted:
                if blocks_after_native[request_id]:
                    raise RuntimeError(
                        "vLLM preempted request retained KV blocks: "
                        f"{request_id!r}")
                preempted_request_ids.append(request_id)
                self._prefix_replay_active[request_id] = False
        cached_request_ids = tuple(
            _vllm_request_id(request_id)
            for request_id in native_output.scheduled_cached_reqs.req_ids)
        resumed_flags = tuple(
            native_output.scheduled_cached_reqs.resumed_from_preemption)
        if len(cached_request_ids) != len(resumed_flags):
            raise RuntimeError(
                "vLLM returned inconsistent resumed-from-preemption metadata")
        if any(not isinstance(resumed, bool) for resumed in resumed_flags):
            raise TypeError(
                "vLLM resumed-from-preemption flags must be boolean")
        resumed_request_ids = tuple(
            request_id
            for request_id, resumed in zip(
                cached_request_ids,
                resumed_flags,
            )
            if resumed
        )
        scheduled_counts = {}
        for request_id, count in native_output.num_scheduled_tokens.items():
            normalized_id = _vllm_request_id(request_id)
            if normalized_id in scheduled_counts:
                raise ValueError(
                    f"vLLM returned duplicate request ID {normalized_id!r}")
            scheduled_counts[normalized_id] = _vllm_integer(
                count, "scheduled token count")
        if not scheduled_counts:
            if self._pending:
                # EngineCore can ask the Scheduler for another batch while an
                # earlier executor future is still outstanding.  An empty
                # decision is the source state machine's signal to wait for
                # the oldest future; it does not consume another slot.
                return None
            raise RuntimeError("vLLM Scheduler returned an empty step while work remains")
        untracked_scheduled_ids = set(scheduled_counts) - set(
            decision_scope_ids)
        if untracked_scheduled_ids:
            raise RuntimeError(
                "vLLM scheduled requests outside the bounded decision scope: "
                f"{sorted(untracked_scheduled_ids)!r}")
        scheduled_spec_tokens = {}
        for request_id, token_ids in (
                native_output.scheduled_spec_decode_tokens.items()):
            normalized_id = _vllm_request_id(request_id)
            if normalized_id in scheduled_spec_tokens:
                raise ValueError(
                    "vLLM returned duplicate speculative request ID "
                    f"{normalized_id!r}")
            if normalized_id not in scheduled_counts:
                raise RuntimeError(
                    "vLLM returned speculative tokens for an unscheduled "
                    f"request: {normalized_id!r}")
            normalized_tokens = tuple(
                _vllm_integer(token_id, "MTP verified token ID")
                for token_id in token_ids
            )
            proposed_tokens = draft_tokens_before_native[normalized_id]
            if normalized_tokens != proposed_tokens[:len(normalized_tokens)]:
                raise RuntimeError(
                    "vLLM scheduled speculative tokens are not a prefix of "
                    f"the installed proposal for {normalized_id!r}")
            scheduled_spec_tokens[normalized_id] = normalized_tokens
        if (self.scheduler_config.mtp_num_speculative_tokens == 0
                and scheduled_spec_tokens):
            raise RuntimeError(
                "vLLM scheduled speculative tokens while MTP is disabled")
        if self.scheduler_config.mtp_num_speculative_tokens:
            speculative_config = self.vllm_config.speculative_config
            if speculative_config is None:
                raise RuntimeError("effective vLLM MTP config disappeared")
            draft_model_config = speculative_config.draft_model_config
            if draft_model_config is None:
                raise RuntimeError("effective vLLM MTP draft model disappeared")
            effective_drafter_max_model_len = draft_model_config.max_model_len
            if effective_drafter_max_model_len is None:
                effective_drafter_max_model_len = (
                    self.vllm_config.model_config.max_model_len)
            effective_drafter_max_model_len = _vllm_integer(
                effective_drafter_max_model_len,
                "MTP drafter max model length",
            )
            max_scheduled_sequence_length = max(
                computed_after_native[request_id]
                for request_id in scheduled_counts)
            mtp_drafter_executes = (
                max_scheduled_sequence_length
                + self.scheduler_config.mtp_num_speculative_tokens
                <= effective_drafter_max_model_len
            )
        else:
            mtp_drafter_executes = False
        replay_step = None
        replay = self.scheduler_config.mtp_acceptance_replay
        if replay is not None:
            replay_step = replay.step_for(self._step_index)
            actual_step_id = (
                f"{self.config.pool_name}:vllm:{self._step_index}")
            if replay_step.step_id != actual_step_id:
                raise RuntimeError(
                    "MTP acceptance replay step ID disagrees with the native "
                    f"Scheduler: expected={replay_step.step_id!r}, "
                    f"actual={actual_step_id!r}")
            selected_request_ids = tuple(scheduled_counts)
            if replay_step.selected_request_ids != selected_request_ids:
                raise RuntimeError(
                    "MTP acceptance replay selected request order disagrees "
                    f"with the native Scheduler: "
                    f"expected={replay_step.selected_request_ids}, "
                    f"actual={selected_request_ids}")
            if (replay_step.preempted_request_ids
                    != tuple(preempted_request_ids)):
                raise RuntimeError(
                    "MTP acceptance replay preempted requests disagree with "
                    f"the native Scheduler: "
                    f"expected={replay_step.preempted_request_ids}, "
                    f"actual={tuple(preempted_request_ids)}")
            if replay_step.mtp_drafter_executed != mtp_drafter_executes:
                raise RuntimeError(
                    "MTP acceptance replay drafter execution disagrees with "
                    f"the native Scheduler at step {self._step_index}")
        resumed_set = set(resumed_request_ids)
        scheduled_ids = set(scheduled_counts)
        if not resumed_set.issubset(scheduled_ids):
            raise RuntimeError(
                "vLLM marked an unscheduled request as resumed from preemption")
        for request_id in resumed_request_ids:
            if status_before_native[request_id] != "PREEMPTED":
                raise RuntimeError(
                    "vLLM resumed flag lacks a PREEMPTED source state for "
                    f"{request_id!r}")
            if status_after_native[request_id] != "RUNNING":
                raise RuntimeError(
                    "vLLM resumed request did not return to RUNNING: "
                    f"{request_id!r}")
        transitioned_to_running = {
            request_id
            for request_id in scheduled_counts
            if status_before_native[request_id] == "PREEMPTED"
            and status_after_native[request_id] == "RUNNING"
        }
        if transitioned_to_running != resumed_set:
            raise RuntimeError(
                "vLLM resumed flags disagree with native request states: "
                f"flags={sorted(resumed_set)}, "
                f"states={sorted(transitioned_to_running)}")

        scheduled_requests = []
        generated_tokens = {}
        finished_request_ids = []
        cache_events = [
            self._capacity_event(
                request_id,
                KVCacheAction.RELEASE,
                0,
                len(blocks_before_native[request_id]),
            )
            for request_id in preempted_request_ids
            if blocks_before_native[request_id]
        ]
        for request_id, count in scheduled_counts.items():
            if count <= 0:
                raise ValueError(
                    f"vLLM scheduled non-positive token count for {request_id!r}")
            request = self._requests[request_id]
            if request_id not in self.scheduler.requests:
                raise RuntimeError(f"vLLM lost scheduled request {request_id!r}")
            native_request = self.scheduler.requests[request_id]

            computed_after_schedule = computed_after_native[request_id]
            computed_before_schedule = computed_after_schedule - count
            if computed_before_schedule < 0:
                raise ValueError(
                    f"vLLM computed-token count regressed for {request_id!r}: "
                    f"after={computed_after_schedule}, scheduled={count}")
            if computed_before_schedule != computed_before_native[request_id]:
                raise RuntimeError(
                    "vLLM scheduled-token base disagrees with native request "
                    f"state for {request_id!r}: "
                    f"before={computed_before_native[request_id]}, "
                    f"derived={computed_before_schedule}")
            high_watermark = self._computed_high_watermark[request_id]
            if high_watermark < computed_before_schedule:
                raise RuntimeError(
                    "vLLM computed high-water mark regressed for "
                    f"{request_id!r}: high={high_watermark}, "
                    f"before={computed_before_schedule}")
            recomputed_tokens = max(
                0,
                min(computed_after_schedule, high_watermark)
                - computed_before_schedule,
            )
            prefix_replay_tokens = 0
            if self._prefix_replay_active[request_id]:
                scheduler_prefix_end = self._scheduler_prefix_hit_tokens[
                    request_id]
                trace_prefix_end = self._trace_prefix_hit_tokens[request_id]
                replay_start = max(
                    computed_before_schedule,
                    scheduler_prefix_end,
                )
                replay_end = min(
                    computed_after_schedule,
                    trace_prefix_end,
                )
                prefix_replay_tokens = max(0, replay_end - replay_start)
                if computed_after_schedule >= trace_prefix_end:
                    self._prefix_replay_active[request_id] = False
            prompt_tokens = request.input_tokens
            phase = (
                "prefill" if computed_before_schedule < prompt_tokens else "decode"
            )
            if phase == "prefill":
                input_tokens = min(
                    count, prompt_tokens - computed_before_schedule)
            elif phase == "decode":
                input_tokens = 0
            else:
                raise RuntimeError(f"vLLM produced unknown phase: {phase!r}")
            output_tokens = count - input_tokens
            first_schedule = request_id not in self._first_schedule
            generated_tokens[request_id] = 0

            before_blocks = blocks_before_native[request_id]
            after_blocks = blocks_after_native[request_id]
            if after_blocks[:len(before_blocks)] != before_blocks:
                raise RuntimeError(
                    "vLLM replaced retained KV blocks without preemption for "
                    f"{request_id!r}: before={before_blocks}, "
                    f"after={after_blocks}")
            if len(after_blocks) > len(before_blocks):
                cache_events.append(self._capacity_event(
                    request_id,
                    KVCacheAction.ALLOCATE,
                    len(before_blocks),
                    len(after_blocks),
                ))

            if first_schedule:
                self._first_schedule.add(request_id)
                cached_tokens = min(computed_before_schedule, prompt_tokens)
                if cached_tokens:
                    cache_events.append(
                        self._cache_event(
                            request_id,
                            KVCacheAction.HIT,
                            0,
                            cached_tokens,
                        )
                    )
            if input_tokens:
                miss_start = min(computed_before_schedule, prompt_tokens)
                miss_end = miss_start + input_tokens
                if miss_end > prompt_tokens:
                    raise RuntimeError(
                        "vLLM prefill chunk exceeds the prompt boundary for "
                        f"{request_id!r}: start={miss_start}, end={miss_end}, "
                        f"prompt={prompt_tokens}")
                cache_events.append(
                    self._cache_event(
                        request_id,
                        KVCacheAction.MISS,
                        miss_start,
                        miss_end,
                    )
                )

            generation_boundary = _vllm_integer(
                native_request.num_tokens,
                "request token count before model output",
            )
            proposed_tokens = len(draft_tokens_before_native[request_id])
            verified_tokens = (
                len(scheduled_spec_tokens[request_id])
                if request_id in scheduled_spec_tokens
                else 0
            )
            if verified_tokens > proposed_tokens:
                raise RuntimeError(
                    "vLLM verified more MTP tokens than were proposed for "
                    f"{request_id!r}")
            accepted_tokens = 0
            proposer_tokens = (
                self.scheduler_config.mtp_num_speculative_tokens
                if mtp_drafter_executes else 0)
            retained_proposed_tokens = len(native_request.spec_token_ids)
            next_proposed_tokens = retained_proposed_tokens
            raw_model_output_tokens = 0
            expected_replay_request = None
            if replay_step is not None:
                expected_replay_request = replay_step.request_for(request_id)
                actual_pre_acceptance_identity = (
                    request_id,
                    phase,
                    input_tokens,
                    output_tokens,
                    computed_before_schedule,
                    recomputed_tokens,
                    prefix_replay_tokens,
                    request_id in resumed_set,
                    self._trace_prefix_hit_tokens[request_id]
                    if first_schedule else 0,
                    self._scheduler_prefix_hit_tokens[request_id]
                    if first_schedule else 0,
                    proposed_tokens,
                    verified_tokens,
                    proposer_tokens,
                )
                if (expected_replay_request.pre_acceptance_identity
                        != actual_pre_acceptance_identity):
                    raise RuntimeError(
                        "MTP acceptance replay request identity disagrees "
                        f"with the native Scheduler at step {self._step_index} "
                        f"for {request_id!r}: "
                        f"expected={expected_replay_request.pre_acceptance_identity}, "
                        f"actual={actual_pre_acceptance_identity}")
            # The pinned model runner discards samples while the scheduled
            # sequence length is still behind Request.num_tokens.  This is
            # stricter than the prompt boundary during chunked recomputation.
            if computed_after_schedule >= generation_boundary:
                generated_before = (
                    len(native_request.output_token_ids)
                    + output_placeholders_before_native[request_id]
                )
                if self.config.is_prefill_only:
                    logical_output_limit = min(1, request.output_tokens)
                else:
                    logical_output_limit = request.output_tokens
                native_output_limit = max(1, logical_output_limit)
                native_output_remaining = (
                    native_output_limit - generated_before)
                model_remaining = (
                    self.vllm_config.scheduler_config.max_model_len
                    - native_request.num_tokens
                )
                raw_output_capacity = min(
                    native_output_remaining, model_remaining)
                if raw_output_capacity <= 0:
                    if (not self.scheduler_config.async_scheduling
                            or output_placeholders_before_native[request_id] <= 0
                            or native_output_remaining > 0):
                        raise RuntimeError(
                            "vLLM scheduled a generation boundary with no "
                            f"output capacity for {request_id!r}")
                    # AsyncScheduler may submit one more decode while the
                    # terminal token is still an unresolved output
                    # placeholder.  The model executes that step, but
                    # update_from_output drops its sample after the earlier
                    # future finishes and removes the request.
                    raw_generated_tokens = 1
                    raw_model_output_tokens = raw_generated_tokens
                    generated_tokens[request_id] = 0
                    next_proposed_tokens = 0
                else:
                    if expected_replay_request is not None:
                        accepted_tokens = (
                            expected_replay_request.mtp_accepted_tokens)
                    elif (verified_tokens
                          and self.scheduler_config.mtp_acceptance_policy
                          == "maximum_entropy"):
                        assumption = (
                            self.scheduler_config.mtp_acceptance_assumption)
                        if assumption is None:
                            raise AssertionError(
                                "validated maximum-entropy assumption disappeared")
                        accepted_tokens = assumption.accepted_count(
                            step_id=(
                                f"{self.config.pool_name}:vllm:"
                                f"{self._step_index}"),
                            request_id=request_id,
                            proposed_tokens=verified_tokens,
                        )
                    elif verified_tokens:
                        accepted_tokens = min(
                            self.scheduler_config.mtp_fixed_accepted_tokens,
                            verified_tokens,
                        )
                    raw_generated_tokens = accepted_tokens + 1
                    raw_model_output_tokens = raw_generated_tokens
                    logical_output_remaining = max(
                        0, logical_output_limit - generated_before)
                    generated_tokens[request_id] = min(
                        raw_generated_tokens,
                        logical_output_remaining,
                        model_remaining,
                    )
                    if raw_generated_tokens >= raw_output_capacity:
                        finished_request_ids.append(request_id)
                        next_proposed_tokens = 0
                    elif proposer_tokens:
                        next_proposed_tokens = proposer_tokens
            elif proposer_tokens:
                next_proposed_tokens = proposer_tokens

            if self.scheduler_config.mtp_num_speculative_tokens == 0:
                mtp_proposer_mode = "disabled"
                mtp_source_kind = "disabled"
                mtp_source_id = "disabled"
            elif proposed_tokens or proposer_tokens:
                mtp_proposer_mode = "padded"
                mtp_source_kind = (
                    "replay_observation"
                    if replay_step is not None
                    else "explicit_assumption"
                )
                mtp_source_id = (
                    self.scheduler_config.mtp_acceptance_source_id)
            else:
                mtp_proposer_mode = "disabled"
                mtp_source_kind = "disabled"
                mtp_source_id = "disabled"
            scheduled_requests.append(
                ScheduledRequest(
                    request_id=request_id,
                    phase=phase,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    computed_tokens=computed_before_schedule,
                    recomputed_tokens=recomputed_tokens,
                    prefix_replay_tokens=prefix_replay_tokens,
                    resumed_from_preemption=request_id in resumed_set,
                    trace_prefix_hit_tokens=(
                        self._trace_prefix_hit_tokens[request_id]
                        if first_schedule else 0
                    ),
                    scheduler_prefix_hit_tokens=(
                        self._scheduler_prefix_hit_tokens[request_id]
                        if first_schedule else 0
                    ),
                    raw_model_output_tokens=raw_model_output_tokens,
                    mtp_proposed_tokens=proposed_tokens,
                    mtp_verified_tokens=verified_tokens,
                    mtp_accepted_tokens=accepted_tokens,
                    mtp_proposer_tokens=proposer_tokens,
                    mtp_next_proposed_tokens=next_proposed_tokens,
                    mtp_proposer_mode=mtp_proposer_mode,
                    mtp_acceptance_source_kind=mtp_source_kind,
                    mtp_acceptance_source_id=mtp_source_id,
                )
            )
            self._computed_high_watermark[request_id] = max(
                high_watermark, computed_after_schedule)

        for request_id in finished_request_ids:
            block_count = len(blocks_after_native[request_id])
            if block_count:
                cache_events.append(self._capacity_event(
                    request_id,
                    KVCacheAction.RELEASE,
                    0,
                    block_count,
                ))

        schedule = ScheduleStep(
            step_id=f"{self.config.pool_name}:vllm:{self._step_index}",
            pool_name=self.config.pool_name,
            requests=tuple(scheduled_requests),
            finished_request_ids=tuple(finished_request_ids),
            preempted_request_ids=tuple(preempted_request_ids),
            attention_kv_layout=self.attention_kv_layout,
        )
        engine_submit_host_s, engine_wake_host_s = (
            self._engine_core_host_round_s(schedule))
        step = BackendStep(
            schedule=schedule,
            kv_cache_events=tuple(cache_events),
            generated_tokens=generated_tokens,
            # Native Scheduler elapsed time is never a Forward input. A
            # versioned profile may instead supply deterministic service for
            # the exact Scheduler state emitted by this step.
            scheduler_plan_cpu_s=(
                self._scheduler_host_service_s(schedule, "plan")
            ),
            # Same contract for the EngineCore busy-loop segments no other
            # Host profile owns: an independently measured versioned profile
            # supplies deterministic submit/wake service, never a trace
            # duration.
            engine_submit_host_s=engine_submit_host_s,
            engine_wake_host_s=engine_wake_host_s,
        )
        if replay_step is not None:
            actual_replay_step = mtp_acceptance_step_from_backend_step(
                self._step_index, step)
            if actual_replay_step != replay_step:
                raise RuntimeError(
                    "MTP acceptance replay outcome disagrees with the native "
                    f"Scheduler at step {self._step_index}: "
                    f"expected={replay_step!r}, actual={actual_replay_step!r}")
        self._pending.append((step, native_output))
        self._step_index += 1
        return step

    def complete_step(
        self,
        step: BackendStep,
        virtual_time_s: float,
        *,
        execute_model: bool,
        execute_transfers: bool,
        sample_tokens: bool,
    ) -> float:
        if execute_model or execute_transfers or sample_tokens:
            raise ValueError(
                "the vLLM simulator backend never executes a data plane")
        if not self._pending or self._pending[0][0] is not step:
            raise ValueError("completion does not match the pending vLLM step")
        _, native_output = self._pending[0]
        scheduled_by_id = {
            request.request_id: request
            for request in step.schedule.requests
        }
        present_before_update = {
            request_id
            for request_id in native_output.num_scheduled_tokens
            if request_id in self.scheduler.requests
        }
        for request_id in native_output.num_scheduled_tokens:
            if request_id not in self.scheduler.requests:
                if step.generated_tokens[request_id] != 0:
                    raise RuntimeError(
                        "vLLM lost a pending request with visible output "
                        f"before update: {request_id!r}")
                continue
            native_request = self.scheduler.requests[request_id]
            normalized_request = scheduled_by_id[request_id]
            expected_raw_generated = (
                normalized_request.raw_model_output_tokens)
            actual_emitted = step.generated_tokens[request_id]
            if not 0 <= actual_emitted <= expected_raw_generated:
                raise RuntimeError(
                    "vLLM generated-token contract disagrees with prompt "
                    f"completion for {request_id!r}: "
                    f"raw={expected_raw_generated}, emitted={actual_emitted}")
        raw_generated_tokens = {
            request_id: scheduled_by_id[request_id].raw_model_output_tokens
            for request_id in native_output.num_scheduled_tokens
        }
        runner_output = _fake_model_runner_output(
            native_output,
            raw_generated_tokens,
            self.scheduler_config.dummy_token_id,
        )
        update_started_ns = (
            time.perf_counter_ns()
            if self._service_measurement_samples is not None else None)
        native_results = self.scheduler.update_from_output(
            native_output, runner_output)
        if update_started_ns is not None:
            self._service_measurement_samples["update_ns"].append(
                time.perf_counter_ns() - update_started_ns)
        proposer_depths = {
            request.mtp_proposer_tokens
            for request in step.schedule.requests
        }
        if proposer_depths == {0}:
            mtp_drafter_executed = False
        elif proposer_depths == {
                self.scheduler_config.mtp_num_speculative_tokens}:
            mtp_drafter_executed = True
        else:
            raise RuntimeError(
                "padded vLLM MTP requires one step-global proposer depth: "
                f"{sorted(proposer_depths)}")
        if mtp_drafter_executed:
            from vllm.v1.outputs import DraftTokenIds

            self.scheduler.update_draft_token_ids(DraftTokenIds(
                req_ids=list(native_output.num_scheduled_tokens),
                draft_token_ids=[
                    [self.scheduler_config.dummy_token_id]
                    * scheduled_by_id[request_id].mtp_proposer_tokens
                    for request_id in native_output.num_scheduled_tokens
                ],
            ))
        self._pending.popleft()
        # See plan_step(): do not feed local wall-clock measurement noise
        # back into the virtual scheduler timeline.
        update_cpu_s = self._scheduler_host_service_s(step.schedule, "update")
        expected_finished = set(step.finished_request_ids)
        released = {
            request_id for request_id in present_before_update
            if request_id not in self.scheduler.requests
        }
        output_finished = {
            output.request_id
            for client_results in native_results.values()
            for output in client_results.outputs
            if output.finish_reason is not None
        }
        emitted_tokens = {
            request_id: 0 for request_id in step.generated_tokens
        }
        for client_results in native_results.values():
            for output in client_results.outputs:
                if output.request_id not in emitted_tokens:
                    raise RuntimeError(
                        "vLLM emitted output for an unscheduled request: "
                        f"{output.request_id!r}")
                request = self._requests[output.request_id]
                if request.output_tokens == 0:
                    emitted_count = 0
                else:
                    emitted_count = len(output.new_token_ids)
                emitted_tokens[output.request_id] += emitted_count
        if emitted_tokens != step.generated_tokens:
            raise RuntimeError(
                "vLLM emitted-token counts disagree with the planned step: "
                f"planned={step.generated_tokens}, emitted={emitted_tokens}")
        if released != expected_finished or output_finished != expected_finished:
            raise RuntimeError(
                "vLLM finish/release state disagrees with the planned step: "
                f"planned={sorted(expected_finished)}, "
                f"released={sorted(released)}, "
                f"output_finished={sorted(output_finished)}")
        for request_id in native_output.num_scheduled_tokens:
            if request_id in self.scheduler.requests:
                computed_tokens = _vllm_integer(
                    self.scheduler.requests[request_id].num_computed_tokens,
                    "computed token count after model output",
                )
                self._computed_high_watermark[request_id] = max(
                    self._computed_high_watermark[request_id],
                    computed_tokens,
                )
        for request_id in step.finished_request_ids:
            del self._requests[request_id]
            del self._computed_high_watermark[request_id]
            del self._trace_prefix_hit_tokens[request_id]
            del self._scheduler_prefix_hit_tokens[request_id]
            del self._prefix_replay_active[request_id]
        return update_cpu_s

    def measure_scheduler_service(
        self,
        *,
        sample_count: int,
        requests_per_batch: int,
        prompt_tokens: int,
        output_tokens: int,
    ) -> dict[str, tuple[int, ...]]:
        """Measure Scheduler-only service over a synthetic control workload."""

        for name, value in (
                ("sample_count", sample_count),
                ("requests_per_batch", requests_per_batch),
                ("prompt_tokens", prompt_tokens),
                ("output_tokens", output_tokens)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Scheduler microbenchmark {name} must be an integer")
            if value < 1:
                raise ValueError(f"Scheduler microbenchmark {name} must be positive")
        if self._pending:
            raise RuntimeError(
                "Scheduler microbenchmark requires no pending model output")
        if self.has_work():
            raise RuntimeError(
                "Scheduler microbenchmark requires a newly constructed backend")
        if requests_per_batch > self.config.max_requests:
            raise ValueError(
                "Scheduler microbenchmark requests_per_batch exceeds capacity")
        if prompt_tokens > self.config.max_batched_tokens:
            raise ValueError(
                "Scheduler microbenchmark prompt_tokens exceeds batch capacity")

        samples: dict[str, list[int]] = {"schedule_ns": [], "update_ns": []}
        self._service_measurement_samples = samples
        request_index = 0
        try:
            while len(samples["schedule_ns"]) < sample_count:
                if not self.has_work():
                    self.add_requests(tuple(
                        RequestSpec(
                            0.0,
                            f"scheduler-microbench-{request_index + offset}",
                            prompt_tokens,
                            output_tokens,
                        )
                        for offset in range(requests_per_batch)
                    ))
                    request_index += requests_per_batch
                step = self.plan_step(0.0)
                if step is None:
                    raise RuntimeError(
                        "Scheduler microbenchmark failed to produce a step")
                self.complete_step(
                    step,
                    0.0,
                    execute_model=False,
                    execute_transfers=False,
                    sample_tokens=False,
                )
        finally:
            self._service_measurement_samples = None
        if (len(samples["schedule_ns"]) != sample_count
                or len(samples["update_ns"]) != sample_count):
            raise RuntimeError("Scheduler microbenchmark sample counts disagree")
        return {
            name: tuple(values[:sample_count])
            for name, values in samples.items()
        }

    def has_work(self) -> bool:
        if self._pending:
            return True
        if self._deferred_requests:
            return True
        if not callable(self.scheduler.has_unfinished_requests):
            raise TypeError(
                "vLLM scheduler has_unfinished_requests must be callable")
        result = self.scheduler.has_unfinished_requests()
        if not isinstance(result, bool):
            raise TypeError(
                "vLLM scheduler has_unfinished_requests() must return boolean")
        return result

    def assert_mtp_contract_exhausted(self) -> None:
        replay = self.scheduler_config.mtp_acceptance_replay
        if replay is not None:
            if self._pending:
                raise RuntimeError(
                    "cannot close vLLM Scheduler with a pending replay step")
            replay.assert_exhausted(self._step_index)
            if self.has_work():
                raise RuntimeError(
                    "MTP acceptance replay is exhausted while the native "
                    "Scheduler still has unfinished requests")
            _validate_mtp_replay_submitted_requests(
                replay.scheduler_context,
                RequestStream(self._accepted_requests),
            )
        elif self.scheduler_config.mtp_acceptance_policy == "maximum_entropy":
            assumption = self.scheduler_config.mtp_acceptance_assumption
            if assumption is None:
                raise AssertionError(
                    "validated maximum-entropy assumption disappeared")
            _validate_assumption_mtp_submitted_requests(
                assumption,
                self.scheduler_config.decode_admission_contract_sha256,
                RequestStream(self._accepted_requests),
                self.scheduler_config.closed_loop_admission_contract_sha256,
            )
        elif self.scheduler_config.request_stream_sha256 is not None:
            if not self._mtp_request_scope_validated:
                raise RuntimeError(
                    "hash-bound MTP request scope was never "
                    "validated")
            accepted_requests = RequestStream(self._accepted_requests)
            if self.scheduler_config.mtp_acceptance_policy == "fixed_count":
                if (self.config.role == "attention"
                        and self.config.max_requests == 2):
                    _validate_standalone_af_mtp_preemption_requests(
                        accepted_requests)
                _validate_fixed_count_mtp_submitted_requests(
                    self.scheduler_config.request_stream_sha256,
                    accepted_requests)
            else:
                raise AssertionError(
                    "request-scoped MTP policy is not fixed-count")

    def _cache_event(
        self,
        request_id: str,
        action: KVCacheAction,
        start: int,
        end: int,
    ) -> KVCacheEvent:
        return KVCacheEvent(
            request_id=request_id,
            action=action,
            tier="hbm",
            token_start=start,
            token_end=end,
            bytes=(end - start) * self.config.kv_bytes_per_token,
        )

    def _capacity_event(
        self,
        request_id: str,
        action: KVCacheAction,
        start_block: int,
        end_block: int,
    ) -> KVCacheEvent:
        if action not in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE):
            raise ValueError(
                "capacity events support only allocate/release actions")
        if start_block < 0 or end_block <= start_block:
            raise ValueError("capacity event requires a positive block range")
        block_size = self.config.kv_block_size_tokens
        return self._cache_event(
            request_id,
            action,
            start_block * block_size,
            end_block * block_size,
        )

    def _block_ids(self, request_id: str) -> tuple[int, ...]:
        block_groups = self.scheduler.kv_cache_manager.get_blocks(
            request_id).get_block_ids()
        if len(block_groups) != 1:
            raise NotImplementedError(
                "the pinned vLLM scheduler baseline requires one KV cache "
                f"group, got {len(block_groups)} for {request_id!r}")
        block_ids = tuple(block_groups[0])
        if any(isinstance(block_id, bool) or not isinstance(block_id, int)
               for block_id in block_ids):
            raise TypeError("vLLM KV block IDs must be integers")
        if len(block_ids) != len(set(block_ids)):
            raise RuntimeError(
                f"vLLM returned duplicate KV block IDs for {request_id!r}")
        return block_ids


def _build_native_scheduler(
    config: BackendConfig,
    scheduler_config: VLLMSchedulerConfig,
):
    _validate_native_scheduler_scope(config, scheduler_config)
    return _construct_native_scheduler(config, scheduler_config)


def _validate_native_scheduler_scope(
    config: BackendConfig,
    scheduler_config: VLLMSchedulerConfig,
) -> None:
    from src.serving.workload.model_config import load_model_config
    scope_model = load_model_config(scheduler_config.model)
    # The Scheduler target scope is one declarative table
    # (src.serving.deployment.validated_scope).  It pins the named target
    # config identities on every row except the 32-layer DeepSeek-V3 TP4
    # row, and deliberately over-approximates: the PP2/PP4 rows accept every
    # named target model in any role, and the PD EP8 row accepts every named
    # target model; the narrower per-model checks in
    # ``_construct_native_scheduler`` and ``_validate_ep_scheduler_model_scope``
    # fail closed afterwards.
    target_scope = is_validated_target_scheduler_scope(
        scope_model,
        role=config.role,
        tensor_parallel_size=scheduler_config.tensor_parallel_size,
        pipeline_parallel_size=scheduler_config.pipeline_parallel_size,
        expert_parallel=scheduler_config.enable_expert_parallel,
    )
    operator_group = worker_role_operator_group(config.role)
    if operator_group == "ffn":
        raise NotImplementedError(
            "the pinned vLLM Scheduler cannot attach to an FFN-only pool")
    if config.role not in (
            "colocated", "prefill", "decode", "attention",
            "prefill-attention", "decode-attention"):
        raise NotImplementedError(
            "the pinned vLLM Scheduler role has not been validated for the "
            f"current deployment composition: {config.role!r}")
    if (scheduler_config.pipeline_parallel_size != 1
            and "VLLM_PP_LAYER_PARTITION" in os.environ):
        raise NotImplementedError(
            "the current vLLM PP Scheduler control-plane baseline requires "
            "the audited uniform layer partition and rejects "
            "VLLM_PP_LAYER_PARTITION")
    if (scheduler_config.tensor_parallel_size not in (1, 2)
            and not target_scope):
        raise NotImplementedError(
            "the current vLLM Scheduler control-plane baseline supports "
            "only TP=1 or independently validated TP=2")
    if (not target_scope
            and scheduler_config.tensor_parallel_size != 1
            and scheduler_config.pipeline_parallel_size != 1
            and (scheduler_config.tensor_parallel_size,
                 scheduler_config.pipeline_parallel_size) != (2, 2)):
        raise NotImplementedError(
            "the current vLLM Scheduler control-plane baseline combines "
            "only the validated TP=2, PP=2 slice")
    if (not target_scope
            and scheduler_config.enable_expert_parallel and (
            scheduler_config.tensor_parallel_size != 2
            or scheduler_config.pipeline_parallel_size != 1)):
        raise NotImplementedError(
            "the validated vLLM EP Scheduler slice requires TP=2 and PP=1")
    standalone_af_tp2 = (
        config.role == "attention"
        and scheduler_config.tensor_parallel_size == 2
        and scheduler_config.pipeline_parallel_size == 1
        and not scheduler_config.enable_expert_parallel
        and scheduler_config.mtp_num_speculative_tokens == 0
    )
    standalone_af_ep2 = (
        config.role == "attention"
        and scheduler_config.tensor_parallel_size == 2
        and scheduler_config.pipeline_parallel_size == 1
        and scheduler_config.enable_expert_parallel
        and scheduler_config.mtp_num_speculative_tokens == 0
        and config.max_batched_tokens == 8
        and config.max_requests == 1
        and config.num_kv_blocks == 8
        and config.kv_block_size_tokens == 16
        and scheduler_config.max_model_len == 128
    )
    if ((scheduler_config.tensor_parallel_size != 1
         or scheduler_config.pipeline_parallel_size != 1)
            and config.role != "colocated"
            and not standalone_af_tp2
            and not standalone_af_ep2
            and not target_scope):
        raise NotImplementedError(
            "the current vLLM parallel Scheduler control-plane baseline supports "
            "only colocated deployment or the validated standalone AF "
            "Attention TP=2/EP=2 slices")
    if ((scheduler_config.tensor_parallel_size != 1
         or scheduler_config.pipeline_parallel_size != 1)
            and scheduler_config.mtp_num_speculative_tokens != 0
            and not target_scope):
        raise NotImplementedError(
            "the current vLLM MTP baseline requires TP=PP=1")
    if config.kv_block_size_tokens != 16:
        raise NotImplementedError(
            "the pinned CPU-only vLLM Scheduler baseline requires a "
            "16-token KV block size")
    if ((config.is_decode_only or config.role == "attention")
            and scheduler_config.mtp_num_speculative_tokens != 0):
        if config.max_requests != 1:
            if config.is_decode_only:
                if not scheduler_config.authorized_h200_pd_mtp_sweep:
                    raise NotImplementedError(
                        "the seeded decode-only vLLM MTP baseline requires "
                        "max_requests=1")
            else:
                af_preemption_scope = (
                    config.max_requests == 2
                    and config.max_batched_tokens == 64
                    and config.num_kv_blocks == 6
                    and scheduler_config.max_model_len == 128
                    and scheduler_config.mtp_num_speculative_tokens == 2
                    and scheduler_config.mtp_acceptance_policy == "fixed_count"
                    and scheduler_config.mtp_fixed_accepted_tokens == 1
                    and scheduler_config.mtp_acceptance_replay is None
                    and scheduler_config.request_stream_sha256 is not None
                )
                if not af_preemption_scope:
                    raise NotImplementedError(
                        "the standalone AF attention vLLM MTP baseline requires "
                        "either "
                        "max_requests=1 or the exact hash-bound AF preemption "
                        "scope")
        if config.is_decode_only:
            if scheduler_config.mtp_acceptance_policy not in (
                    "fixed_count", "maximum_entropy", "replay"):
                raise NotImplementedError(
                    "the seeded decode-only vLLM MTP baseline supports only "
                    "fixed_count, maximum_entropy, or replay acceptance")
            replay = scheduler_config.mtp_acceptance_replay
            if (scheduler_config.mtp_acceptance_policy == "replay"
                    and (replay is None
                         or replay.schema_version != 2
                         or replay.scheduler_context.adapter_contract
                         != PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT)):
                raise NotImplementedError(
                    "decode-only vLLM MTP replay requires the PD decode "
                    "replay schema v2 adapter contract")
        elif (scheduler_config.mtp_acceptance_policy != "fixed_count"
              or scheduler_config.mtp_acceptance_replay is not None):
            raise NotImplementedError(
                "standalone AF attention vLLM MTP currently requires "
                "fixed_count acceptance without a replay artifact")
    elif (config.role != "colocated"
          and scheduler_config.mtp_num_speculative_tokens != 0):
        raise NotImplementedError(
            "non-colocated vLLM MTP is validated only for the decode role "
            "or standalone AF attention role")


def _validate_ep_scheduler_model_scope(
    scheduler_config: VLLMSchedulerConfig,
    model,
) -> None:
    if not scheduler_config.enable_expert_parallel:
        return
    if model.model_type in TARGET_MODEL_PRODUCTION_PARALLELISM:
        target_parallel = TARGET_MODEL_PRODUCTION_PARALLELISM[model.model_type]
    elif model.model_type in ("qwen3", "qwen3_moe", "llama", "glm4_moe"):
        target_parallel = None
    else:
        raise AssertionError(
            f"unhandled Scheduler model type: {model.model_type!r}")
    if target_parallel is not None and (
            scheduler_config.tensor_parallel_size,
            scheduler_config.pipeline_parallel_size) in (target_parallel, (8, 1)):
        return
    if not is_validated_qwen3_moe_ep2_model_scope(model):
        raise NotImplementedError(
            "the validated vLLM EP Scheduler slice requires BF16 Qwen3MoE "
            "with an even expert count, valid top-k, all-sparse decoder "
            "layers, and no shared experts")


def _validate_tp2_scheduler_model_scope(
    scheduler_config: VLLMSchedulerConfig,
    model,
) -> None:
    if (scheduler_config.tensor_parallel_size != 2
            or scheduler_config.pipeline_parallel_size != 1
            or scheduler_config.enable_expert_parallel):
        return
    if model.model_type == "qwen3":
        if model.compute_dtype == "bf16":
            scheduler_dtype = "bfloat16"
        elif model.compute_dtype == "fp16":
            scheduler_dtype = "float16"
        else:
            raise NotImplementedError(
                f"unsupported Qwen3 compute dtype: {model.compute_dtype!r}"
            )
        supported_model = (
            model.compute_dtype in ("bf16", "fp16")
            and model.weight_dtype == model.compute_dtype
            and scheduler_config.dtype == scheduler_dtype
        )
    elif model.model_type == "llama":
        supported_model = (
            is_validated_llama_tp2_model_scope(model)
            and model.compute_dtype == "bf16"
            and model.weight_dtype == "bf16"
            and scheduler_config.dtype == "bfloat16"
        )
    else:
        raise NotImplementedError(
            f"unsupported TP2 Scheduler model type: {model.model_type!r}"
        )
    if (not supported_model
            or model.num_experts != 0
            or model.attention_layout != "standard_gqa"):
        raise NotImplementedError(
            "the validated vLLM TP=2 Scheduler slice requires dense BF16 or "
            "FP16 Qwen3 with matching model and Scheduler dtypes, or the "
            "exact source-backed Llama fixture in BF16, with standard GQA and "
            "expert parallelism disabled")


def _validate_tp2_pp2_scheduler_model_scope(
    scheduler_config: VLLMSchedulerConfig,
    model,
) -> None:
    if (scheduler_config.tensor_parallel_size,
            scheduler_config.pipeline_parallel_size) != (2, 2):
        return
    if (scheduler_config.enable_expert_parallel
            or model.model_type != "qwen3"
            or model.num_experts != 0
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"
            or scheduler_config.dtype != "bfloat16"):
        raise NotImplementedError(
            "the validated vLLM TP=2, PP=2 Scheduler slice requires dense "
            "BF16 Qwen3 with standard GQA and expert parallelism disabled")


def _validate_tp1_pp_scheduler_model_scope(
    scheduler_config: VLLMSchedulerConfig,
    model,
) -> None:
    pipeline_size = scheduler_config.pipeline_parallel_size
    if (pipeline_size == 1
            or scheduler_config.tensor_parallel_size != 1):
        return
    if model.model_type == "kimi_k2":
        target_pipeline_size = 16
    elif model.model_type == "glm_moe_dsa":
        target_pipeline_size = 32
    elif model.model_type in ("qwen3", "qwen3_moe", "llama", "glm4_moe"):
        target_pipeline_size = None
    else:
        raise AssertionError(
            f"unhandled Scheduler model type: {model.model_type!r}")
    if target_pipeline_size is not None:
        if (pipeline_size != target_pipeline_size
                or scheduler_config.enable_expert_parallel
                or model.compute_dtype != "bf16"
                or scheduler_config.dtype != "bfloat16"
                or scheduler_config.mtp_num_speculative_tokens != 0):
            raise NotImplementedError(
                "target Scheduler requires Kimi-K2.6 TP1/PP16 or "
                "GLM-5.2 TP1/PP32 with EP and MTP disabled")
        return
    if (scheduler_config.tensor_parallel_size != 1
            or scheduler_config.enable_expert_parallel
            or model.model_type != "qwen3"
            or model.num_experts != 0
            or model.tie_word_embeddings
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"
            or scheduler_config.dtype != "bfloat16"):
        raise NotImplementedError(
            "the validated vLLM TP=1 uniform PP Scheduler path requires "
            "untied dense BF16 Qwen3 with standard GQA and expert "
            "parallelism disabled")
    if model.num_hidden_layers % pipeline_size:
        raise NotImplementedError(
            "the validated vLLM TP=1 uniform PP Scheduler path requires "
            "model layers must be divisible by pipeline parallel size: "
            f"layers={model.num_hidden_layers}, PP={pipeline_size}")


def _construct_native_scheduler(
    config: BackendConfig,
    scheduler_config: VLLMSchedulerConfig,
):
    from src.serving.backend.vllm_scheduler_process import (
        _WORKER_ENVIRONMENT_FLAG,
    )
    import os

    if (_WORKER_ENVIRONMENT_FLAG not in os.environ
            or os.environ[_WORKER_ENVIRONMENT_FLAG] != "1"):
        raise RuntimeError(
            "native vLLM Scheduler construction is restricted to the "
            "isolated CPU worker")
    try:
        import torch
        from vllm.config import (
            CacheConfig,
            ParallelConfig,
            SchedulerConfig,
            StructuredOutputsConfig,
        )
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            MLAAttentionSpec,
        )
        from vllm.v1.structured_output import StructuredOutputManager
    except ImportError as exc:
        raise RuntimeError(
            "vLLMBackend.from_model requires a vLLM v1 installation"
        ) from exc

    from src.serving.workload.model_config import load_model_config

    model = load_model_config(scheduler_config.model)
    _validate_ep_scheduler_model_scope(scheduler_config, model)
    _validate_tp2_scheduler_model_scope(scheduler_config, model)
    _validate_tp2_pp2_scheduler_model_scope(scheduler_config, model)
    _validate_tp1_pp_scheduler_model_scope(scheduler_config, model)
    if scheduler_config.dtype == "bfloat16":
        requested_compute_dtype = "bf16"
        torch_dtype = torch.bfloat16
    elif scheduler_config.dtype == "float16":
        requested_compute_dtype = "fp16"
        torch_dtype = torch.float16
    else:
        raise AssertionError(
            f"unhandled validated vLLM dtype: {scheduler_config.dtype!r}")
    if model.compute_dtype != requested_compute_dtype:
        raise ValueError(
            "vLLM requested dtype disagrees with the canonical model "
            f"config: requested={scheduler_config.dtype!r}, "
            f"model={model.compute_dtype!r}")

    effective_max_model_len = scheduler_config.max_model_len
    if effective_max_model_len is None:
        effective_max_model_len = model.max_position_embeddings
    if effective_max_model_len > model.max_position_embeddings:
        raise ValueError(
            "vLLM max_model_len exceeds the canonical model limit: "
            f"requested={effective_max_model_len}, "
            f"model={model.max_position_embeddings}")

    from vllm.platforms import current_platform
    if not current_platform.is_cpu():
        raise RuntimeError(
            "isolated vLLM Scheduler worker did not select CpuPlatform")

    if (scheduler_config.mtp_num_speculative_tokens
            and model.mtp_architecture not in (
                "glm4_moe_mtp", "glm_moe_dsa_mtp", "deepseek_mtp")):
        raise NotImplementedError(
            "the pinned vLLM MTP bridge requires a source-backed GLM or "
            "DeepSeek-V3 MTP architecture")
    native_mtp_engine_config = (
        scheduler_config.mtp_num_speculative_tokens > 0
        and model.mtp_architecture == "glm4_moe_mtp")
    if native_mtp_engine_config:
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model=scheduler_config.model,
            tensor_parallel_size=scheduler_config.tensor_parallel_size,
            pipeline_parallel_size=scheduler_config.pipeline_parallel_size,
            dtype=scheduler_config.dtype,
            max_model_len=effective_max_model_len,
            gpu_memory_utilization=scheduler_config.gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=scheduler_config.trust_remote_code,
            enable_chunked_prefill=scheduler_config.enable_chunked_prefill,
            enable_prefix_caching=False,
            max_num_batched_tokens=config.max_batched_tokens,
            max_num_seqs=config.max_requests,
            block_size=config.kv_block_size_tokens,
            load_format="auto",
            skip_tokenizer_init=True,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": (
                    scheduler_config.mtp_num_speculative_tokens),
            },
        )
        vllm_config = engine_args.create_engine_config()
        native_model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        num_layers = native_model_config.get_num_layers(parallel_config)
        layer_start = 0
        layer_end = num_layers
        num_kv_heads = native_model_config.get_num_kv_heads(parallel_config)
        head_size = native_model_config.get_head_size()
        cache_dtype = native_model_config.dtype
        kv_spec_layout = "standard_gqa"
    else:
        if scheduler_config.trust_remote_code:
            raise NotImplementedError(
                "scheduler-only construction never executes remote model "
                "code; trust_remote_code must be false")
        if model.attention_layout == "standard_gqa":
            if model.weight_dtype != model.compute_dtype:
                raise NotImplementedError(
                    "scheduler-only standard-GQA construction does not yet "
                    "support mixed or quantized weight storage")
        elif model.attention_layout == "mla":
            if model.model_type == "kimi_k2":
                if (model.architecture != "DeepseekV3ForCausalLM"
                        or model.compute_dtype != "bf16"
                        or scheduler_config.dtype != "bfloat16"
                        or (scheduler_config.tensor_parallel_size,
                            scheduler_config.pipeline_parallel_size) not in (
                            (1, 1), (8, 2), (8, 1))
                        or scheduler_config.enable_expert_parallel
                        != (scheduler_config.tensor_parallel_size == 8)
                        or model.num_mtp_layers != 0):
                    raise NotImplementedError(
                        "the validated scheduler-only MLA slice supports exactly "
                        "Kimi-K2.6 BF16 compute with TP1/PP1, TP=EP=8/PP2, or "
                        "TP=EP=8/PP1, and MTP disabled")
            elif model.model_type == "deepseek_v3":
                dsv3_parallel_scope = (
                    (scheduler_config.tensor_parallel_size,
                     scheduler_config.pipeline_parallel_size,
                     scheduler_config.enable_expert_parallel,
                     config.role)
                    in (
                        (1, 1, False, "colocated"),
                        (8, 1, False, "colocated"),
                        (8, 1, False, "prefill"),
                        (8, 1, False, "decode"),
                    )
                    or (
                        # 32-layer local variant measured on 4x H200.
                        model.config_sha256 == DEEPSEEK_V3_L32_CONFIG_SHA256
                        and (scheduler_config.tensor_parallel_size,
                             scheduler_config.pipeline_parallel_size,
                             scheduler_config.enable_expert_parallel,
                             config.role) == (4, 1, False, "colocated")
                    )
                )
                if (model.architecture != "DeepseekV3ForCausalLM"
                        or model.compute_dtype != "bf16"
                        or model.weight_dtype != "fp8"
                        or scheduler_config.dtype != "bfloat16"
                        or not dsv3_parallel_scope
                        or model.mtp_architecture != "deepseek_mtp"
                        or model.num_mtp_layers != 1):
                    raise NotImplementedError(
                        "the validated scheduler-only MLA slice supports exactly "
                        "DeepSeek-V3 FP8 with TP1/PP1 colocated or H200 "
                        "TP8/PP1/EP1 colocated/prefill/decode, and its "
                        "single source-backed MTP layer")
            else:
                raise AssertionError(
                    f"unhandled MLA Scheduler model type: "
                    f"{model.model_type!r}")
        elif model.attention_layout == "dsa_mla":
            dsa_parallel_scope = (
                (scheduler_config.tensor_parallel_size,
                 scheduler_config.pipeline_parallel_size,
                 scheduler_config.enable_expert_parallel,
                 config.role)
                in (
                    (1, 1, False, "colocated"),
                    (8, 4, True, "colocated"),
                    (8, 1, False, "colocated"),
                    (8, 1, True, "prefill"),
                    (8, 1, True, "decode"),
                    (8, 1, False, "prefill"),
                    (8, 1, False, "decode"),
                )
            )
            if (model.model_type != "glm_moe_dsa"
                    or model.architecture != "GlmMoeDsaForCausalLM"
                    or model.compute_dtype != "bf16"
                    or model.weight_dtype not in ("bf16", "fp8")
                    or scheduler_config.dtype != "bfloat16"
                    or not dsa_parallel_scope
                    or (scheduler_config.mtp_num_speculative_tokens > 0
                        and (model.mtp_architecture != "glm_moe_dsa_mtp"
                             or model.num_mtp_layers != 1))):
                raise NotImplementedError(
                    "the validated scheduler-only DSA slice supports exactly "
                    "GLM-5.2 BF16 with TP1/PP1 colocated, TP=EP=8/PP4 "
                    "colocated, H200 aggregate TP8/PP1/EP1 colocated, or "
                    "H200 PD roles TP8/PP1 with EP8 or EP1, and its single "
                    "source-backed MTP layer")
        else:
            raise AssertionError(
                f"unhandled attention layout: {model.attention_layout!r}")
        native_scheduler_config = SchedulerConfig(
            runner_type="generate",
            max_num_batched_tokens=config.max_batched_tokens,
            max_num_seqs=config.max_requests,
            max_model_len=effective_max_model_len,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=0,
            enable_chunked_prefill=(
                scheduler_config.enable_chunked_prefill),
            is_multimodal_model=False,
            policy="fcfs",
            num_lookahead_slots=(
                scheduler_config.mtp_num_speculative_tokens),
            disable_chunked_mm_input=False,
            disable_hybrid_kv_cache_manager=False,
            async_scheduling=scheduler_config.async_scheduling,
        )
        cache_config = CacheConfig(
            block_size=config.kv_block_size_tokens,
            gpu_memory_utilization=scheduler_config.gpu_memory_utilization,
            cache_dtype="auto",
            is_attention_free=False,
            sliding_window=None,
            enable_prefix_caching=False,
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=scheduler_config.tensor_parallel_size,
            pipeline_parallel_size=scheduler_config.pipeline_parallel_size,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
            enable_expert_parallel=(
                scheduler_config.enable_expert_parallel),
            enable_eplb=False,
            distributed_executor_backend="uni",
            decode_context_parallel_size=1,
        )
        if (scheduler_config.pipeline_parallel_size > 1
                and model.model_type not in ("kimi_k2", "glm_moe_dsa")):
            from vllm.engine.arg_utils import EngineArgs

            native_model_config = EngineArgs(
                model=scheduler_config.model,
                tensor_parallel_size=scheduler_config.tensor_parallel_size,
                pipeline_parallel_size=(
                    scheduler_config.pipeline_parallel_size),
                dtype=scheduler_config.dtype,
                max_model_len=effective_max_model_len,
                trust_remote_code=False,
                load_format="auto",
                skip_tokenizer_init=True,
            ).create_model_config()
            native_model_config.verify_with_parallel_config(parallel_config)
        if (model.num_attention_heads
                % scheduler_config.tensor_parallel_size):
            raise ValueError(
                "model attention heads must be divisible by tensor parallel "
                "size: "
                f"{model.num_attention_heads} vs "
                f"{scheduler_config.tensor_parallel_size}")
        if (model.attention_layout == "standard_gqa"
                and model.num_hidden_layers
                % scheduler_config.pipeline_parallel_size):
            raise ValueError(
                "model layers must be divisible by pipeline parallel size: "
                f"{model.num_hidden_layers} vs "
                f"{scheduler_config.pipeline_parallel_size}")
        layer_ranges = pipeline_layer_ranges_for_model(
            model, scheduler_config.pipeline_parallel_size)
        representative_stage = max(
            range(len(layer_ranges)),
            key=lambda stage: model.kv_bytes_per_token_for_layer_range(
                scheduler_config.tensor_parallel_size,
                *layer_ranges[stage],
            ),
        )
        layer_start, layer_end = layer_ranges[representative_stage]
        num_layers = layer_end - layer_start
        model_view = _SchedulerModelView(
            max_model_len=effective_max_model_len,
            vocab_size=model.vocab_size,
            dtype=torch_dtype,
        )
        if scheduler_config.mtp_num_speculative_tokens:
            speculative_config = _SchedulerSpeculativeConfig(
                method="mtp",
                num_speculative_tokens=(
                    scheduler_config.mtp_num_speculative_tokens),
                draft_model_config=_SchedulerDraftModelConfig(
                    hf_config=_SchedulerDraftHFConfig(
                        model_type=model.mtp_architecture,
                        num_nextn_predict_layers=model.num_mtp_layers,
                    ),
                    max_model_len=effective_max_model_len,
                ),
            )
        else:
            speculative_config = None
        vllm_config = _SchedulerOnlyVllmConfig(
            model_config=model_view,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=native_scheduler_config,
            structured_outputs_config=StructuredOutputsConfig(),
            kv_geometry_representative_pipeline_stage_rank=(
                representative_stage),
            kv_geometry_representative_layer_start=layer_start,
            kv_geometry_representative_layer_end=layer_end,
            speculative_config=speculative_config,
        )
        if model.attention_layout == "standard_gqa":
            num_kv_heads = model.kv_heads_per_tensor_parallel_rank(
                scheduler_config.tensor_parallel_size)
            head_size = model.head_dim
            kv_spec_layout = "standard_gqa"
        elif model.attention_layout == "mla":
            num_kv_heads = 1
            head_size = model.kv_lora_rank + model.qk_rope_head_dim
            kv_spec_layout = "mla"
        elif model.attention_layout == "dsa_mla":
            # The pinned control-plane Scheduler only consumes block counts.
            # Represent DSA as one manager group whose page is the exact sum
            # of every compressed-MLA and full-indexer cache page; the tensor
            # list below still preserves the real per-layer ownership/bytes.
            dsa_bytes_per_token = (
                model.kv_bytes_per_token_for_layer_range(
                    scheduler_config.tensor_parallel_size,
                    layer_start,
                    layer_end,
                ))
            if dsa_bytes_per_token % model.dtype_bytes:
                raise RuntimeError(
                    "GLM-5.2 DSA KV bytes must be representable by the "
                    "Scheduler aggregate cache dtype")
            num_kv_heads = 1
            head_size = dsa_bytes_per_token // model.dtype_bytes
            kv_spec_layout = "dsa_aggregate"
        else:
            raise AssertionError(
                "validated scheduler layout changed before KV construction")
        cache_dtype = torch_dtype

    if not isinstance(vllm_config.model_config.dtype, torch.dtype):
        raise TypeError(
            "vLLM model_config.dtype must be a torch.dtype, got "
            f"{type(vllm_config.model_config.dtype).__name__}")
    if kv_spec_layout == "standard_gqa":
        cache_spec = FullAttentionSpec(
            block_size=config.kv_block_size_tokens,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=cache_dtype,
        )
    elif kv_spec_layout in ("mla", "dsa_aggregate"):
        cache_spec = MLAAttentionSpec(
            block_size=config.kv_block_size_tokens,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=cache_dtype,
            cache_dtype_str=None,
        )
    else:
        raise AssertionError(
            f"unhandled Scheduler KV spec layout: {kv_spec_layout!r}")
    if model.model_type in ("kimi_k2", "glm_moe_dsa", "deepseek_v3"):
        layer_names = [
            f"model.layers.{index}.self_attn.attn"
            for index in range(layer_start, layer_end)
        ]
    elif model.model_type in ("qwen3", "qwen3_moe", "llama", "glm4_moe"):
        layer_names = [
            f"model.layers.{index}.self_attn.attn"
            for index in range(num_layers)
        ]
    else:
        raise AssertionError(
            f"unhandled Scheduler model type: {model.model_type!r}")
    page_bytes = cache_spec.page_size_bytes
    if kv_spec_layout == "dsa_aggregate":
        compressed_page_bytes = (
            config.kv_block_size_tokens
            * (model.kv_lora_rank + model.qk_rope_head_dim)
            * model.dtype_bytes
        )
        index_page_bytes = config.kv_block_size_tokens * (
            model.index_head_dim + model.index_head_dim // 128 * 4)
        dsa_tensor_page_bytes = []
        for indexer_type in model.indexer_types[layer_start:layer_end]:
            if indexer_type == "full":
                layer_page_bytes = (
                    compressed_page_bytes + index_page_bytes)
            elif indexer_type == "shared":
                layer_page_bytes = compressed_page_bytes
            else:
                raise AssertionError(
                    f"unhandled GLM-5.2 indexer type: {indexer_type!r}")
            dsa_tensor_page_bytes.append(layer_page_bytes)
        tensor_page_bytes = tuple(dsa_tensor_page_bytes)
        if sum(tensor_page_bytes) != page_bytes:
            raise RuntimeError(
                "GLM-5.2 per-layer DSA cache tensors disagree with the "
                "aggregate Scheduler cache spec")
    elif kv_spec_layout in ("standard_gqa", "mla"):
        tensor_page_bytes = (page_bytes,) * len(layer_names)
    else:
        raise AssertionError(
            f"unhandled Scheduler tensor layout: {kv_spec_layout!r}")
    kv_cache_config = KVCacheConfig(
        num_blocks=config.num_kv_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=config.num_kv_blocks * layer_page_bytes,
                shared_by=[layer_name],
            )
            for layer_name, layer_page_bytes in zip(
                layer_names, tensor_page_bytes, strict=True)
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=cache_spec)
        ],
    )
    actual_tensor_owners = tuple(
        tuple(tensor.shared_by)
        for tensor in kv_cache_config.kv_cache_tensors
    )
    expected_tensor_owners = tuple((layer_name,) for layer_name in layer_names)
    if actual_tensor_owners != expected_tensor_owners:
        raise RuntimeError(
            "vLLM Scheduler KV tensors must be owned one per layer: "
            f"actual={actual_tensor_owners!r}, "
            f"expected={expected_tensor_owners!r}")
    if page_bytes % config.kv_block_size_tokens:
        raise RuntimeError(
            "vLLM Scheduler KV page bytes must divide by block tokens")
    total_tensor_bytes = sum(
        tensor.size for tensor in kv_cache_config.kv_cache_tensors)
    capacity_denominator = (
        config.num_kv_blocks * config.kv_block_size_tokens)
    if total_tensor_bytes % capacity_denominator:
        raise RuntimeError(
            "vLLM Scheduler KV tensor bytes must divide by its block "
            "capacity")
    constructed_kv_bytes_per_token = (
        total_tensor_bytes // capacity_denominator)
    model_kv_bytes_per_token = model.kv_bytes_per_token_for_layer_range(
        scheduler_config.tensor_parallel_size,
        layer_start,
        layer_end,
    )
    if constructed_kv_bytes_per_token != model_kv_bytes_per_token:
        raise RuntimeError(
            "vLLM Scheduler KV spec disagrees with ModelConfig capacity: "
            f"constructed={constructed_kv_bytes_per_token}, "
            f"model={model_kv_bytes_per_token}")
    if config.kv_bytes_per_token != model_kv_bytes_per_token:
        raise RuntimeError(
            "BackendConfig KV bytes disagree with the representative "
            "maximum-KV-geometry pipeline stage: "
            f"backend={config.kv_bytes_per_token}, "
            f"model={model_kv_bytes_per_token}")
    vllm_config.cache_config.num_gpu_blocks = config.num_kv_blocks
    vllm_config.cache_config.num_gpu_blocks_override = config.num_kv_blocks
    _validate_effective_scheduler_config(
        vllm_config,
        config,
        scheduler_config,
    )
    scheduler_class = (
        AsyncScheduler if scheduler_config.async_scheduling else Scheduler
    )
    scheduler = scheduler_class(
        vllm_config,
        kv_cache_config,
        StructuredOutputManager(vllm_config),
    )
    return scheduler, vllm_config


def _validate_effective_scheduler_config(
    vllm_config: Any,
    config: BackendConfig,
    requested: VLLMSchedulerConfig,
) -> None:
    """Fail if pinned vLLM rewrites a scheduler limit outside our contract."""

    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    if requested.enable_chunked_prefill is not True:
        raise RuntimeError("requested vLLM chunked prefill must be true")
    if scheduler.enable_chunked_prefill is not True:
        raise RuntimeError("vLLM effective enable_chunked_prefill is not true")
    if scheduler.chunked_prefill_enabled is not True:
        raise RuntimeError("vLLM effective chunked_prefill_enabled is not true")
    if scheduler.max_num_batched_tokens != config.max_batched_tokens:
        raise RuntimeError(
            "vLLM rewrote max_num_batched_tokens: "
            f"requested={config.max_batched_tokens}, "
            f"effective={scheduler.max_num_batched_tokens}")
    if scheduler.max_num_seqs != config.max_requests:
        raise RuntimeError(
            "vLLM rewrote max_num_seqs: "
            f"requested={config.max_requests}, "
            f"effective={scheduler.max_num_seqs}")
    expected_max_model_len = requested.max_model_len
    if expected_max_model_len is None:
        expected_max_model_len = vllm_config.model_config.max_model_len
    if scheduler.max_model_len != expected_max_model_len:
        raise RuntimeError(
            "vLLM rewrote max_model_len: "
            f"expected={expected_max_model_len}, "
            f"effective={scheduler.max_model_len}")
    if scheduler.policy != "fcfs":
        raise RuntimeError(
            f"vLLM scheduler policy must be 'fcfs', got {scheduler.policy!r}")
    if (scheduler.max_num_partial_prefills != 1
            or scheduler.max_long_partial_prefills != 1
            or scheduler.long_prefill_token_threshold != 0):
        raise RuntimeError(
            "vLLM partial-prefill policy differs from the pinned baseline")
    if scheduler.disable_chunked_mm_input is not False:
        raise RuntimeError(
            "vLLM chunked multimodal input must remain disabled")
    if scheduler.async_scheduling is not requested.async_scheduling:
        raise RuntimeError(
            "vLLM effective async scheduling disagrees with the requested "
            "Scheduler control-plane mode")
    expected_lookahead = requested.mtp_num_speculative_tokens
    if scheduler.num_lookahead_slots != expected_lookahead:
        raise RuntimeError(
            "vLLM speculative lookahead disagrees with MTP config: "
            f"requested={expected_lookahead}, "
            f"effective={scheduler.num_lookahead_slots}")
    speculative = vllm_config.speculative_config
    if requested.mtp_num_speculative_tokens == 0:
        if speculative is not None:
            raise RuntimeError("vLLM speculative decoding must be disabled")
    else:
        if speculative is None:
            raise RuntimeError("vLLM dropped the requested MTP configuration")
        if speculative.method != "mtp":
            raise RuntimeError(
                "vLLM speculative method must remain 'mtp', got "
                f"{speculative.method!r}")
        if (speculative.num_speculative_tokens
                != requested.mtp_num_speculative_tokens):
            raise RuntimeError(
                "vLLM rewrote the requested MTP proposal count")
        if speculative.disable_padded_drafter_batch is not False:
            raise RuntimeError(
                "pinned vLLM MTP workload requires padded drafter batches")
        draft_model_config = speculative.draft_model_config
        if draft_model_config is None:
            raise RuntimeError("vLLM MTP requires an effective draft model")
        draft_hf_config = draft_model_config.hf_config
        if draft_hf_config.model_type not in (
                "glm4_moe_mtp", "glm_moe_dsa_mtp", "deepseek_mtp"):
            raise NotImplementedError(
                "Serving MTP workload requires a source-backed GLM or "
                "DeepSeek-V3 MTP architecture")
        if draft_hf_config.num_nextn_predict_layers != 1:
            raise NotImplementedError(
                "Serving GLM/DeepSeek MTP workload currently supports one "
                "MTP layer")
        if draft_hf_config.model_type == "glm4_moe_mtp":
            if draft_hf_config.num_hidden_layers != 0:
                raise RuntimeError(
                    "vLLM GLM4 MTP draft config must start at decoder layer "
                    "zero")
        elif draft_hf_config.model_type in ("glm_moe_dsa_mtp", "deepseek_mtp"):
            if not isinstance(
                    draft_model_config, _SchedulerDraftModelConfig):
                raise RuntimeError(
                    "GLM-5.2/DeepSeek-V3 MTP must use the scheduler-only "
                    "draft contract")
        else:
            raise AssertionError("validated MTP draft architecture changed")
        effective_draft_max_model_len = draft_model_config.max_model_len
        if effective_draft_max_model_len is None:
            effective_draft_max_model_len = vllm_config.model_config.max_model_len
        if (isinstance(effective_draft_max_model_len, bool)
                or not isinstance(effective_draft_max_model_len, int)
                or effective_draft_max_model_len <= 0):
            raise RuntimeError(
                "vLLM MTP draft max model length must be a positive integer")
    if vllm_config.kv_transfer_config is not None:
        raise RuntimeError("vLLM native KV transfer must be disabled")
    if cache.enable_prefix_caching is not False:
        raise RuntimeError("vLLM content prefix caching must be disabled")
    if cache.block_size != config.kv_block_size_tokens:
        raise RuntimeError(
            "vLLM rewrote KV block size: "
            f"requested={config.kv_block_size_tokens}, "
            f"effective={cache.block_size}")
    if cache.gpu_memory_utilization != requested.gpu_memory_utilization:
        raise RuntimeError(
            "vLLM rewrote gpu_memory_utilization: "
            f"requested={requested.gpu_memory_utilization}, "
            f"effective={cache.gpu_memory_utilization}")
    if cache.num_gpu_blocks != config.num_kv_blocks:
        raise RuntimeError(
            "vLLM effective KV block count disagrees with BackendConfig")
    if cache.num_gpu_blocks_override != config.num_kv_blocks:
        raise RuntimeError(
            "vLLM KV block override disagrees with BackendConfig")
    if cache.cache_dtype != "auto":
        raise RuntimeError("vLLM KV cache dtype must remain model-derived")
    if cache.is_attention_free is not False:
        raise RuntimeError("vLLM text baseline requires attention KV cache")
    if cache.sliding_window is not None:
        raise NotImplementedError(
            "the vLLM scheduler baseline does not support sliding-window KV")
    expected_world_size = (
        requested.tensor_parallel_size * requested.pipeline_parallel_size)
    if (parallel.tensor_parallel_size != requested.tensor_parallel_size
            or parallel.pipeline_parallel_size
            != requested.pipeline_parallel_size
            or parallel.data_parallel_size != 1
            or parallel.data_parallel_rank != 0
            or parallel.decode_context_parallel_size != 1
            or parallel.world_size != expected_world_size):
        raise RuntimeError(
            "vLLM effective parallel config disagrees with the requested "
            "Scheduler control-plane scope")
    if (parallel.enable_expert_parallel
            is not requested.enable_expert_parallel):
        raise RuntimeError(
            "vLLM effective expert-parallel enablement disagrees with the "
            "requested Scheduler control-plane scope")


def _make_native_request(
    vllm_config: Any,
    request: RequestSpec,
    *,
    dummy_token_id: int,
    max_output_tokens: int | None = None,
):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    dummy_token_id = _vllm_integer(dummy_token_id, "dummy_token_id")
    if not hasattr(vllm_config.model_config, "get_vocab_size"):
        raise TypeError("vLLM model_config.get_vocab_size must be callable")
    get_vocab_size = vllm_config.model_config.get_vocab_size
    if not callable(get_vocab_size):
        raise TypeError("vLLM model_config.get_vocab_size must be callable")
    vocab_size = _vllm_integer(
        get_vocab_size(),
        "model vocab_size",
    )
    if vocab_size <= 0:
        raise ValueError("vLLM model vocab_size must be positive")
    if not 0 <= dummy_token_id < vocab_size:
        raise ValueError(
            "vLLM dummy_token_id must be within the model vocabulary")
    if max_output_tokens is not None:
        max_output_tokens = _vllm_integer(
            max_output_tokens, "max_output_tokens")
        if max_output_tokens <= 0:
            raise ValueError("vLLM max_output_tokens must be positive")
    output_limit = (
        request.output_tokens
        if max_output_tokens is None
        else max_output_tokens
    )
    return Request(
        request_id=request.request_id,
        prompt_token_ids=_LengthOnlyTokenIds(
            dummy_token_id, request.input_tokens),
        sampling_params=SamplingParams(
            max_tokens=output_limit),
        pooling_params=None,
        eos_token_id=None,
        arrival_time=request.arrival_time_s,
        block_hasher=None,
    )


def _verify_content_hash_state_disabled(native_request: Any) -> None:
    """Fail if a synthetic length-only request acquires content identity."""

    if native_request.block_hashes != []:
        raise RuntimeError(
            "length-only vLLM request unexpectedly contains block hashes")
    if native_request.get_hash_new_full_blocks is not None:
        raise RuntimeError(
            "length-only vLLM request unexpectedly enables content hashing")


def _fake_model_runner_output(
    native_output: Any,
    generated_tokens: dict[str, int],
    dummy_token_id: int,
):
    from vllm.v1.outputs import ModelRunnerOutput

    request_ids = list(native_output.num_scheduled_tokens)
    if tuple(generated_tokens) != tuple(request_ids):
        raise ValueError(
            "generated_tokens must preserve native vLLM request order: "
            f"native={request_ids}, normalized={list(generated_tokens)}")
    scheduled_spec_tokens = native_output.scheduled_spec_decode_tokens
    sampled_token_ids = []
    for request_id in request_ids:
        generated = _vllm_integer(
            generated_tokens[request_id], "generated token count")
        draft_token_ids = (
            list(scheduled_spec_tokens[request_id])
            if request_id in scheduled_spec_tokens
            else []
        )
        if draft_token_ids:
            if not 1 <= generated <= len(draft_token_ids) + 1:
                raise ValueError(
                    "speculative vLLM output must contain an accepted draft "
                    "prefix followed by one terminal token")
            sampled_token_ids.append(
                draft_token_ids[:generated - 1] + [dummy_token_id])
        elif generated in (0, 1):
            sampled_token_ids.append([dummy_token_id] * generated)
        else:
            raise ValueError(
                "non-speculative vLLM steps must generate zero or one token")
    return ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={request_id: index for index, request_id in enumerate(request_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
