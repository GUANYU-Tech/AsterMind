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
from dataclasses import asdict, dataclass
from itertools import islice, repeat
import math
import os
from pathlib import Path
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
from fractions import Fraction

from src.serving.workload.model_config import (
    DEEPSEEK_V3_L32_CONFIG_SHA256,
    DEEPSEEK_V4_FLASH_CONFIG_SHA256,
    DEEPSEEK_V4_PRO_CONFIG_SHA256,
    GLM_BLOCK_FP8_CONFIG_SHA256S,
    KIMI_K2_7_CODE_CONFIG_SHA256,
    KIMI_K3_CONFIG_SHA256,
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


def _scheduled_ordinary_token_ids(
    token_ids: Sequence[int], *, computed_tokens: int, rows: int,
    num_tokens: int, pending_outputs: int, allow_pending_outputs: bool,
) -> tuple[int, ...] | None:
    """Keep native async output placeholders unknown, never replace IDs."""
    if len(token_ids) != num_tokens:
        raise RuntimeError("native token prefix length disagrees with num_tokens")
    ordinary = tuple(_vllm_integer(token_id, "scheduled length-only input token ID")
        for token_id in token_ids[computed_tokens:computed_tokens + rows])
    if len(ordinary) == rows:
        return ordinary
    # vLLM0.11 AsyncScheduler schedules num_tokens + output placeholders,
    # while Request.all_token_ids contains only the resolved contiguous prefix.
    # The pre-schedule placeholder count, not the increment for this step,
    # bounds the missing input suffix. Existing IDs remain validated above.
    if (allow_pending_outputs and pending_outputs > 0
            and computed_tokens >= 0 and rows > 0
            and computed_tokens + rows <= num_tokens + pending_outputs):
        return None
    raise RuntimeError("native scheduled token prefix does not cover the declared query")


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
    prefix_cache_mode: str = "trace"
    async_scheduling: bool = False
    dummy_token_id: int = 42
    mtp_num_speculative_tokens: int = 0
    mtp_acceptance_policy: str = "disabled"
    mtp_fixed_accepted_tokens: int = 0
    mtp_acceptance_source_id: str = "disabled"
    mtp_acceptance_replay: MTPAcceptanceReplay | None = None
    mtp_draft_token_ids_by_step: Mapping[str, Mapping[str, tuple[int, ...]]] | None = None
    mtp_acceptance_assumption: MTPAcceptanceAssumption | None = None
    request_stream_sha256: str | None = None
    decode_admission_contract_sha256: str | None = None
    closed_loop_admission_contract_sha256: str | None = None
    authorized_h200_pd_mtp_sweep: bool = False
    scheduler_host_profile: SchedulerHostProfile | None = None
    engine_core_host_round_profile: EngineCoreHostRoundProfile | None = None
    scheduler_version: str = "0.11.0"
    export_embedding_token_ids: bool = False

    def __post_init__(self) -> None:
        if self.prefix_cache_mode not in ("trace", "token_ids"):
            raise ValueError("prefix_cache_mode must be trace or token_ids")
        if self.prefix_cache_mode == "token_ids":
            if not self.enable_prefix_caching:
                raise ValueError("token_ids mode requires enable_prefix_caching")
            if self.scheduler_version != "0.11.0":
                raise NotImplementedError("token prefix cache currently requires the native0.11 Request hashing contract")
            if self.async_scheduling:
                raise NotImplementedError("token prefix cache does not support async scheduling")
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
        if self.mtp_draft_token_ids_by_step is not None:
            if self.prefix_cache_mode != "token_ids" or not self.mtp_num_speculative_tokens:
                raise ValueError("explicit MTP draft IDs require token_ids mode and MTP")
            if not isinstance(self.mtp_draft_token_ids_by_step, Mapping):
                raise TypeError("MTP draft frames must be a step/request mapping")
            frames = {}
            for step_id, rows in self.mtp_draft_token_ids_by_step.items():
                if not isinstance(step_id, str) or not step_id or not isinstance(rows, Mapping):
                    raise ValueError("MTP draft frames require nonempty step IDs and request mappings")
                frame = {}
                for request_id, tokens in rows.items():
                    if not isinstance(request_id, str) or not request_id:
                        raise ValueError("MTP draft frames require nonempty request IDs")
                    if (not isinstance(tokens, tuple) or len(tokens) != self.mtp_num_speculative_tokens
                            or any(type(token) is not int or token < 0 for token in tokens)):
                        raise ValueError("MTP draft IDs must be nonnegative integer tuples covering proposer depth")
                    frame[request_id] = tokens
                frames[step_id] = frame
            # Plain copied dictionaries preserve multiprocessing spawn/pickle.
            object.__setattr__(self, "mtp_draft_token_ids_by_step", frames)
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
        if self.scheduler_version not in ("0.11.0", "0.30.0"):
            raise ValueError("vLLM scheduler_version must be 0.11.0 or 0.30.0")
        if self.scheduler_version == "0.11.0" and not self.enable_chunked_prefill:
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

    @classmethod
    def from_vllm030_scheduler(
        cls,
        config: BackendConfig,
        scheduler_config: VLLMSchedulerConfig,
        scheduler: Any,
        *,
        source_root: Path,
        worker_kv_configs: tuple[Any, ...],
    ) -> "VLLMBackend":
        """Bind a prepared real 0.30 CPU Scheduler to the simulation protocol.

        The caller owns process isolation, device guards and cache allocation.
        This entry verifies source/config/cache geometry, but cannot establish
        available-KV provenance, H200 workers or model/sampler execution. Model
        feedback remains the explicit length-only simulation used by this owner.
        """
        if not isinstance(config, BackendConfig) or not isinstance(scheduler_config, VLLMSchedulerConfig):
            raise TypeError("0.30 bridge requires BackendConfig and VLLMSchedulerConfig")
        if scheduler_config.scheduler_version != "0.30.0":
            raise ValueError("0.30 bridge requires explicit scheduler_version=0.30.0")
        if ((config.role not in ("colocated", "prefill", "decode")
                and worker_role_operator_group(config.role) != "attention") or config.scheduler_pipeline_depth != 1
                or (scheduler_config.pipeline_parallel_size != 1 and config.role != "colocated")
                or scheduler_config.async_scheduling
                or scheduler_config.enable_prefix_caching
                or scheduler_config.dtype != "bfloat16"):
            raise NotImplementedError("0.30 prepared bridge requires whole-model synchronous PP1/MTP-off or bound GLM depth2 without prefix caching")
        if (scheduler_config.scheduler_host_profile is not None
                or scheduler_config.engine_core_host_round_profile is not None):
            raise NotImplementedError("0.30 bridge has no bound version-specific Host service profile")
        from src.serving.backend.vllm_scheduler_process import vllm030_source_identity
        from src.serving.kv_cache.vllm_allocation import (
            normalize_vllm030_model_allocation,
            vllm030_uses_plain_mla_cache,
            vllm030_supports_expert_parallel_cache,
            vllm030_worker_cache_specs,
            validate_vllm030_cache_placement,
        )
        from src.serving.workload.model_config import load_model_config
        identity = vllm030_source_identity(source_root)
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, KVCacheConfig, UniformTypeKVCacheSpecs
        from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

        if type(scheduler) is not Scheduler:
            raise TypeError("0.30 bridge requires the real synchronous upstream Scheduler")
        if scheduler.get_num_unfinished_requests() != 0:
            raise ValueError("0.30 prepared bridge requires an unused Scheduler")
        native = scheduler.vllm_config
        if native.device_config.device_type != "cpu":
            raise NotImplementedError("0.30 prepared bridge currently requires a guarded CPU process")
        model = load_model_config(scheduler_config.model)
        glm_pd_source = (model.model_type == "glm_moe_dsa"
            and model.model_id in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3")
            and scheduler_config.tensor_parallel_size == 8
            and scheduler_config.mtp_num_speculative_tokens == 0)
        kimi_pd_source = (model.model_id == "moonshotai/Kimi-K2.6"
            and scheduler_config.enable_expert_parallel
            and vllm030_supports_expert_parallel_cache(model, scheduler_config.tensor_parallel_size)
            and scheduler_config.mtp_num_speculative_tokens == 0)
        if (config.role in ("prefill", "decode")
                and model.model_type not in ("qwen3", "qwen3_moe") and not (glm_pd_source or kimi_pd_source)):
            raise NotImplementedError("0.30 metadata PD bridge requires Qwen3, GLM TP8 EP1/8 or K2.6 EP8 MTP-off pools")
        if worker_role_operator_group(config.role) == "attention" and (
                model.model_type not in ("qwen3", "llama", "qwen3_moe")
                or scheduler_config.tensor_parallel_size not in ((1, 2) if config.role == "attention" else (1,))
                or scheduler_config.enable_expert_parallel):
            raise NotImplementedError("0.30 AF metadata bridge requires ordinary Qwen3/Qwen3MoE/Llama TP1/2 attention")
        if ((model.model_id not in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3",
                                   "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro",
                                   "moonshotai/Kimi-K2.6") and model.model_type not in ("qwen3", "qwen3_moe", "llama")
                and not vllm030_uses_plain_mla_cache(model))
                or load_model_config(native.model_config.model).config_sha256 != model.config_sha256):
            raise ValueError("0.30 bridge requires matching accurate source model configs")
        if scheduler_config.enable_expert_parallel and not vllm030_supports_expert_parallel_cache(
                model, scheduler_config.tensor_parallel_size):
            raise NotImplementedError("0.30 prepared bridge requires Qwen3MoE EP2 or GLM/K2.6 EP8 cache scope")
        if scheduler_config.mtp_num_speculative_tokens:
            from src.serving.kv_cache.vllm_allocation import validate_vllm030_glm_mtp_config
            validate_vllm030_glm_mtp_config(native, model)
            if (config.role != "colocated" or scheduler_config.pipeline_parallel_size != 1
                    or scheduler_config.mtp_num_speculative_tokens != 2
                    or scheduler_config.mtp_acceptance_policy != "fixed_count"
                    or native.speculative_config.num_speculative_tokens != scheduler_config.mtp_num_speculative_tokens):
                raise NotImplementedError("0.30 MTP bridge requires colocated GLM depth2 and explicit fixed-count acceptance")
        elif native.speculative_config is not None:
            raise ValueError("0.30 native Scheduler unexpectedly enables speculation")
        if (native.kv_transfer_config is not None
                or native.parallel_config.pipeline_parallel_size != scheduler_config.pipeline_parallel_size
                or native.parallel_config.tensor_parallel_size != scheduler_config.tensor_parallel_size
                or native.parallel_config.enable_expert_parallel != scheduler_config.enable_expert_parallel):
            raise ValueError("0.30 native Scheduler mode disagrees with bridge scope")
        for actual, expected, name in (
                (scheduler.max_num_scheduled_tokens, config.max_batched_tokens, "scheduled batch tokens"),
                (native.scheduler_config.max_num_seqs, config.max_requests, "request limit"),
                (native.model_config.max_model_len, scheduler_config.max_model_len, "model length"),
                (native.scheduler_config.enable_chunked_prefill, scheduler_config.enable_chunked_prefill, "chunked prefill"),
                (native.scheduler_config.async_scheduling, False, "async scheduling"),
                (native.cache_config.enable_prefix_caching, False, "prefix caching"),
                (native.cache_config.gpu_memory_utilization, scheduler_config.gpu_memory_utilization, "memory utilization"),
                (native.cache_config.block_size, config.kv_block_size_tokens, "block size"),
                (scheduler.kv_cache_config.num_blocks, config.num_kv_blocks, "block count")):
            if actual != expected:
                raise ValueError(f"0.30 effective {name} disagrees with bridge declaration")
        if native.cache_config.num_gpu_blocks_override is not None:
            raise ValueError("0.30 bridge rejects debug block overrides")
        rank_specs = vllm030_worker_cache_specs(native, model)
        expected_specs = rank_specs[0]
        if (not isinstance(worker_kv_configs, tuple)
                or len(worker_kv_configs) != len(rank_specs)
                or any(not isinstance(worker, KVCacheConfig) for worker in worker_kv_configs)):
            raise TypeError("0.30 bridge requires the complete native worker cache configurations")
        arena_bytes_by_rank = []
        for rank, (worker, expected) in enumerate(zip(worker_kv_configs, rank_specs, strict=True)):
            if worker.num_blocks != config.num_kv_blocks:
                raise ValueError("0.30 native worker block count disagrees with bridge")
            arena_bytes_by_rank.append(validate_vllm030_cache_placement(native, worker))
            tp = scheduler_config.tensor_parallel_size
            if worker.kv_cache_groups != worker_kv_configs[rank // tp * tp].kv_cache_groups:
                raise ValueError("0.30 PP1 worker cache groups disagree across TP ranks")
            actual_specs = {}
            for group in worker.kv_cache_groups:
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                    members = group.kv_cache_spec.kv_cache_specs
                else:
                    members = {name: group.kv_cache_spec for name in group.layer_names}
                if set(members) != set(group.layer_names) or set(actual_specs).intersection(members):
                    raise ValueError("0.30 cache group has inconsistent or duplicate owners")
                actual_specs.update(members)
            if actual_specs != expected:
                raise ValueError("0.30 native worker cache specs disagree with the complete source-projected cache map")
        if scheduler.kv_cache_config != generate_scheduler_kv_cache_config(list(worker_kv_configs)):
            raise ValueError("0.30 Scheduler cache representative disagrees with native worker configs")
        allocation = normalize_vllm030_model_allocation(
            native, model, worker_kv_configs, scheduler.kv_cache_config,
            source_root=source_root)
        page_bytes = sum(spec.page_size_bytes for spec in expected_specs.values())
        if model.model_type == "deepseek_v4":
            reporting_rate = model.kv_bytes_per_token_per_parallel_rank(scheduler_config.tensor_parallel_size, 1)
            if config.kv_bytes_per_token != -(-reporting_rate.numerator // reporting_rate.denominator):
                raise ValueError("0.30 V4 bridge nominal KV reporting bytes disagree with model geometry")
            # All groups alias the same physical pool. A live block in any
            # group reserves the complete pool row, including padding.
            if any(arena % allocation.num_kv_blocks for arena in arena_bytes_by_rank):
                raise ValueError("0.30 V4 arena does not have integral physical pool rows")
            row_bytes = arena_bytes_by_rank[0] // allocation.num_kv_blocks
            if any(arena // allocation.num_kv_blocks != row_bytes for arena in arena_bytes_by_rank):
                raise ValueError("0.30 V4 physical pool rows disagree across TP ranks")
        elif page_bytes != config.kv_bytes_per_token * config.kv_block_size_tokens:
            raise ValueError("0.30 bridge KV bytes disagree with source cache geometry")
        # Upstream deliberately reduces each uniform worker group to its first
        # spec for Scheduler block management. Traffic must retain the full
        # worker group page, including all main and indexer states.
        geometry = tuple((group.kv_cache_spec.block_size,
                          row_bytes if model.model_type == "deepseek_v4" else
                          sum(expected_specs[name].page_size_bytes for name in group.layer_names)
                          if type(group.kv_cache_spec) in (FullAttentionSpec, MLAAttentionSpec) else group.kv_cache_spec.page_size_bytes)
                         for group in worker_kv_configs[0].kv_cache_groups)
        control_plane = _NativeVLLMControlPlane(
            config, scheduler_config, prepared_scheduler=scheduler,
            prepared_kv_geometry=geometry, prepared_model=model)
        control_plane.runtime_info = {
            **identity, "platform": "guarded CPU, not H200 runtime",
            "capacity_source": "caller-prepared native allocation; capacity provenance unverified",
            "cache_specs_source": "Midlayer source projection, not constructed model layers",
            "cache_geometry_source": "complete worker groups; Scheduler representative does not carry aggregate traffic",
            "cache_placement_source_validated": True,
            "native_allocation": asdict(allocation),
            "kv_group_geometry": geometry,
            "shared_cache_arena_bytes_by_rank": tuple(arena_bytes_by_rank),
            "model_feedback_source": "synthetic length-only simulator token feedback; not model sampling",
            "effective_chunked_prefill": native.scheduler_config.enable_chunked_prefill,
            "effective_tensor_parallel_size": native.parallel_config.tensor_parallel_size,
            "effective_expert_parallel": native.parallel_config.enable_expert_parallel,
            "effective_sequence_parallel_moe": native.parallel_config.use_sequence_parallel_moe,
            "effective_max_num_batched_tokens": native.scheduler_config.max_num_batched_tokens,
            "effective_max_num_scheduled_tokens": scheduler.max_num_scheduled_tokens,
            "effective_cache_block_size": native.cache_config.block_size,
            "normal_capacity_accepted": False, "runtime_acceptance": False,
            "mtp_scenario": mtp_scenario_manifest(scheduler_config),
            "speculative_method": "disabled" if native.speculative_config is None else native.speculative_config.method,
            "mtp_num_speculative_tokens": 0 if native.speculative_config is None else native.speculative_config.num_speculative_tokens,
            "mtp_draft_architecture": (None if native.speculative_config is None else
                native.speculative_config.draft_model_config.hf_config.architectures[0]),
        }
        return cls(config, control_plane)

    @classmethod
    def from_vllm030_capacity_plan(
        cls, capacity_plan, scheduler_config: VLLMSchedulerConfig, *,
        python_executable: Path, source_root: Path, max_batched_tokens: int, max_requests: int,
        supported_layouts_by_rank: tuple[tuple[str, ...], ...],
    ) -> "VLLMBackend":
        """Bind an analytical normal-capacity plan to its real 0.30 process.

        Reconcile actual returned allocation facts with the planner before
        admitting any request. This does not upgrade policy/reserve provenance
        or CPU control-plane execution to H200 runtime acceptance.
        """
        from src.serving.kv_cache.kv_capacity import KVCapacityPlan
        from src.serving.workload.model_config import load_model_config
        from src.serving.kv_cache.vllm_allocation import vllm030_supports_expert_parallel_cache

        if not isinstance(capacity_plan, KVCapacityPlan) or not isinstance(scheduler_config, VLLMSchedulerConfig):
            raise TypeError("0.30 analytical backend requires a capacity plan and Scheduler config")
        if (capacity_plan.source_mode != "analytical" or capacity_plan.native_allocation is None
                or capacity_plan.native_allocation.version != "0.30.0"
                or scheduler_config.scheduler_version != "0.30.0"):
            raise ValueError("0.30 analytical backend requires native0.30 analytical facts, not debug capacity")
        if (capacity_plan.model_config_sha256 != load_model_config(scheduler_config.model).config_sha256
                or capacity_plan.tensor_parallel_size != scheduler_config.tensor_parallel_size
                or capacity_plan.pipeline_parallel_size != scheduler_config.pipeline_parallel_size
                or (capacity_plan.worker_role not in ("colocated", "prefill", "decode")
                    and worker_role_operator_group(capacity_plan.worker_role) != "attention")
                or (scheduler_config.enable_expert_parallel and not vllm030_supports_expert_parallel_cache(
                    load_model_config(scheduler_config.model), scheduler_config.tensor_parallel_size))):
            raise ValueError("0.30 analytical backend model/parallelism disagrees with capacity plan")
        backend = cls.from_vllm030_byte_budget(
            scheduler_config, python_executable=python_executable, source_root=source_root,
            pool_name=capacity_plan.pool_name, max_batched_tokens=max_batched_tokens, max_requests=max_requests,
            role=capacity_plan.worker_role,
            kv_block_size_tokens=capacity_plan.kv_block_size_tokens,
            available_memory_by_rank=tuple(budget.available_kv_bytes for budget in capacity_plan.rank_budgets),
            supported_layouts_by_rank=supported_layouts_by_rank)
        if (backend.config.max_batched_tokens != max_batched_tokens or backend.config.max_requests != max_requests
                or backend.runtime_info["native_allocation"] != asdict(capacity_plan.native_allocation)):
            backend.close()
            raise ValueError("0.30 effective allocation or limits disagree with analytical capacity plan")
        return backend

    @classmethod
    def from_vllm030_byte_budget(
        cls, scheduler_config: VLLMSchedulerConfig, *, python_executable: Path,
        source_root: Path, pool_name: str, max_batched_tokens: int, max_requests: int,
        kv_block_size_tokens: int, available_memory_by_rank: tuple[int, ...],
        supported_layouts_by_rank: tuple[tuple[str, ...], ...], role: str = "colocated",
    ) -> "VLLMBackend":
        """Start isolated 0.30 control-plane simulation on declared KV budgets.

        This is distinct from the normal-capacity ``from_model`` path. It
        records upstream CPU rewrites and explicit synthetic model feedback;
        the inputs do not establish H200 memory, provider or model execution.
        """
        from src.serving.backend.vllm_scheduler_process import VLLM030SchedulerProcess
        control = VLLM030SchedulerProcess(
            scheduler_config, python_executable=python_executable, source_root=source_root,
            pool_name=pool_name, max_batched_tokens=max_batched_tokens,
            role=role,
            max_requests=max_requests, kv_block_size_tokens=kv_block_size_tokens,
            available_memory_by_rank=available_memory_by_rank,
            supported_layouts_by_rank=supported_layouts_by_rank)
        return cls(control.backend_config, control)

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
    def __init__(self, config: BackendConfig, scheduler_config: VLLMSchedulerConfig,
                 *, prepared_scheduler: Any = None,
                 prepared_kv_geometry: tuple[tuple[int, int], ...] | None = None,
                 prepared_model: Any = None) -> None:
        self.config = config
        self.scheduler_config = scheduler_config
        self.scheduler_host_profile = scheduler_config.scheduler_host_profile
        self.engine_core_host_round_profile = (
            scheduler_config.engine_core_host_round_profile)
        self._scheduler_host_uncovered_reported: set[str] = set()
        if prepared_scheduler is None:
            self.scheduler, self.vllm_config, dsv4_capacity_context = (
                _build_native_scheduler(config, scheduler_config))
        else:
            if scheduler_config.scheduler_version != "0.30.0":
                raise ValueError("prepared Scheduler requires explicit 0.30 bridge")
            self.scheduler = prepared_scheduler
            self.vllm_config = prepared_scheduler.vllm_config
            dsv4_capacity_context = None
        if scheduler_config.prefix_cache_mode == "token_ids":
            from vllm.v1.core.kv_cache_utils import init_none_hash
            from vllm.utils import get_hash_fn_by_name
            init_none_hash(get_hash_fn_by_name(self.vllm_config.cache_config.prefix_caching_hash_algo))
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MLAAttentionSpec,
        )
        kv_groups = self.scheduler.kv_cache_config.kv_cache_groups
        self._prepared_v4 = prepared_model is not None and prepared_model.model_type == "deepseek_v4"
        self._prepared_mla_token_ids = prepared_model is not None and prepared_model.model_type in ("glm_moe_dsa", "kimi_k2", "deepseek_v3")
        if dsv4_capacity_context is not None:
            # DeepSeek-V4 runs one Scheduler group per cache lifecycle class
            # (review-06 R1); the attention layout remains the MLA-style
            # compressed contract.
            self._dsv4_groups, self._dsv4_model = dsv4_capacity_context
            self.attention_kv_layout = "mla"
            if len(kv_groups) != len(self._dsv4_groups):
                raise RuntimeError(
                    "DeepSeek-V4 Scheduler group count disagrees with the "
                    f"ModelConfig cache groups: scheduler={len(kv_groups)}, "
                    f"model={len(self._dsv4_groups)}")
        elif self._prepared_v4:
            self._dsv4_groups = None
            self._dsv4_model = prepared_model
            self.attention_kv_layout = "mla"
        else:
            self._dsv4_groups = None
            self._dsv4_model = None
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
        if scheduler_config.scheduler_version == "0.30.0":
            if prepared_kv_geometry is None or len(prepared_kv_geometry) != len(kv_groups):
                raise ValueError("0.30 bridge requires complete worker cache traffic geometry")
            self._kv_group_geometry = prepared_kv_geometry
            from vllm.v1.kv_cache_interface import SlidingWindowMLASpec, UniformTypeKVCacheSpecs
            sliding = []
            for group in kv_groups:
                spec = group.kv_cache_spec
                members = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
                flags = {isinstance(member, SlidingWindowMLASpec) for member in members}
                if len(flags) != 1:
                    raise ValueError("0.30 cache group mixes full and sliding lifecycles")
                sliding.append(next(iter(flags)))
            self._sliding_cache_groups = tuple(sliding)
        else:
            self._kv_group_geometry = tuple(
                (group.kv_cache_spec.block_size,
                 group.kv_cache_spec.page_size_bytes)
                for group in kv_groups)
            if self._dsv4_groups is None:
                # A uniform Scheduler group shares block IDs across layers;
                # its spec page is per layer (or already aggregate for DSA).
                # Capacity events charge the actual worker tensor rows once.
                native_cache = self.scheduler.kv_cache_config
                row_bytes, remainder = divmod(sum(tensor.size for tensor in native_cache.kv_cache_tensors),
                    native_cache.num_blocks)
                if remainder or row_bytes <= 0:
                    raise ValueError("0.11 worker KV tensors must contain whole positive cache rows")
                group, = kv_groups
                self._kv_group_geometry = ((group.kv_cache_spec.block_size, row_bytes),)
            self._sliding_cache_groups = (tuple(group.lifecycle == "sliding_window" for group in self._dsv4_groups)
                if self._dsv4_groups is not None else (False,) * len(kv_groups))
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
        if self.scheduler_config.scheduler_version == "0.30.0" and request.prefix_hit_tokens:
            raise NotImplementedError("0.30 prepared bridge has no bound prefix-oracle replay contract")
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
            scheduler_version=self.scheduler_config.scheduler_version,
        )
        if self.scheduler_config.prefix_cache_mode == "trace":
            _verify_content_hash_state_disabled(native, self.scheduler_config.scheduler_version)
        if self.config.is_decode_only:
            if request.prefix_hit_tokens != 0:
                raise NotImplementedError(
                    "decode-only vLLM pools cannot re-inject a trace prefix hit")
            # Metadata-only PD connector: the prefill pool has already produced
            # the first token and the transfer edge has delivered all prompt KV.
            # Seeding these two pieces of native Request state makes vLLM
            # allocate/account the transferred blocks and schedule decode only.
            native.num_computed_tokens = request.input_tokens
            if self.scheduler_config.prefix_cache_mode == "token_ids":
                if request.output_tokens < 2:
                    raise ValueError("decode-only admission requires remaining output after the transferred first token")
                native.append_output_token_ids(request.output_token_ids[0])
            else:
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
                if any(blocks_after_native[request_id]):
                    raise RuntimeError(
                        "vLLM preempted request retained KV blocks: "
                        f"{request_id!r}")
                preempted_request_ids.append(request_id)
                self._prefix_replay_active[request_id] = False
        cached_request_ids = tuple(
            _vllm_request_id(request_id)
            for request_id in native_output.scheduled_cached_reqs.req_ids)
        resumed_request_ids = _resumed_native_request_ids(
            native_output.scheduled_cached_reqs, cached_request_ids,
            self.scheduler_config.scheduler_version)
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
            event
            for request_id in preempted_request_ids
            for event in self._capacity_events(
                request_id,
                KVCacheAction.RELEASE,
                blocks_before_native[request_id],
                blocks_after_native[request_id],
            )
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
            content_hit = 0
            if (self.scheduler_config.prefix_cache_mode == "token_ids"
                    and not self.config.is_decode_only
                    and status_before_native[request_id] in ("WAITING", "PREEMPTED")):
                content_hit = (computed_before_schedule
                               if status_before_native[request_id] == "WAITING" else 0)
                self._computed_high_watermark[request_id] = max(
                    self._computed_high_watermark[request_id], computed_before_schedule)
            if (computed_before_schedule != computed_before_native[request_id]
                    and not (self.scheduler_config.prefix_cache_mode == "token_ids"
                             and computed_before_native[request_id] == 0)):
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
            for group_index, (before_group, after_group) in enumerate(
                    zip(before_blocks, after_blocks, strict=True)):
                if self._sliding_cache_groups[group_index]:
                    # Windowed caches legitimately free blocks as the window
                    # slides (SlidingWindowManager.remove_skipped_blocks);
                    # that is the review-06 R1 lifecycle, not preemption.
                    continue
                if after_group[:len(before_group)] != before_group:
                    raise RuntimeError(
                        "vLLM replaced retained KV blocks without preemption "
                        f"for {request_id!r}: before={before_blocks}, "
                        f"after={after_blocks}")
            if self._prepared_v4:
                cache_events.extend(self._capacity_events(
                    request_id, KVCacheAction.RELEASE, before_blocks, after_blocks))
            cache_events.extend(self._capacity_events(
                request_id,
                KVCacheAction.ALLOCATE,
                before_blocks,
                after_blocks,
            ))

            if first_schedule:
                self._first_schedule.add(request_id)
                cached_tokens = min(computed_before_schedule, prompt_tokens)
                if cached_tokens and not (self.config.is_decode_only
                        and self.scheduler_config.prefix_cache_mode == "token_ids"):
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
                    (self.vllm_config.model_config.max_model_len
                     if self.scheduler_config.scheduler_version == "0.30.0"
                     else self.vllm_config.scheduler_config.max_model_len)
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
                    content_prefix_hit_tokens=content_hit,
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
            cache_events.extend(self._capacity_events(
                request_id,
                KVCacheAction.RELEASE,
                blocks_after_native[request_id],
                tuple(() for _ in blocks_after_native[request_id]),
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
        embedding_ids = None
        if (self.scheduler_config.export_embedding_token_ids
                or self._prepared_v4 or self._prepared_mla_token_ids
                or self.attention_kv_layout == "paged_varlen"
                or self.scheduler_config.prefix_cache_mode == "token_ids"):
            scheduled_ids = []
            unresolved_input_ids = False
            for request in schedule.requests:
                # Scheduler Request.all_token_ids excludes the speculative
                # suffix. GPUModelRunner appends the actual scheduled draft
                # IDs after the non-speculative prefix for each query.
                ordinary_rows = request.scheduled_tokens - request.mtp_verified_tokens
                native_request = self.scheduler.requests[request.request_id]
                ordinary = _scheduled_ordinary_token_ids(
                    native_request.all_token_ids,
                    computed_tokens=request.computed_tokens, rows=ordinary_rows,
                    num_tokens=_vllm_integer(native_request.num_tokens, "native token count"),
                    pending_outputs=output_placeholders_before_native[request.request_id],
                    allow_pending_outputs=(self.scheduler_config.scheduler_version == "0.11.0"
                        and self.scheduler_config.async_scheduling
                        and self.config.scheduler_pipeline_depth > 1
                        and self.scheduler_config.mtp_num_speculative_tokens == 0
                        and request.mtp_verified_tokens == 0))
                if ordinary is None:
                    unresolved_input_ids = True
                else:
                    scheduled_ids.extend(ordinary)
                if request.mtp_verified_tokens:
                    drafts = scheduled_spec_tokens[request.request_id]
                    if len(drafts) != request.mtp_verified_tokens:
                        raise RuntimeError("native scheduled draft IDs do not cover the declared verification rows")
                    scheduled_ids.extend(_vllm_integer(token_id, "scheduled speculative input token ID") for token_id in drafts)
            if not unresolved_input_ids:
                embedding_ids = tuple(scheduled_ids)
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
            embedding_token_ids=embedding_ids,
            embedding_token_source=(("vllm_explicit_request_token_ids"
                if self.scheduler_config.prefix_cache_mode == "token_ids" else "vllm_length_only_simulation")
                if embedding_ids is not None else None),
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
        if self.scheduler_config.prefix_cache_mode == "token_ids":
            for index, request_id in enumerate(runner_output.req_ids):
                request = self._requests[request_id]
                if raw_generated_tokens[request_id] and request.output_tokens:
                    native_request = self.scheduler.requests[request_id]
                    offset = len(native_request.output_token_ids)
                    visible = step.generated_tokens[request_id]
                    real_tokens = list(request.output_token_ids[offset:offset + visible])
                    if len(real_tokens) != visible:
                        raise ValueError("real output transcript does not cover the native generated tokens")
                    accepted = min(scheduled_by_id[request_id].mtp_accepted_tokens, visible)
                    scheduled_drafts = native_output.scheduled_spec_decode_tokens
                    if accepted and tuple(real_tokens[:accepted]) != tuple(scheduled_drafts[request_id][:accepted]):
                        raise ValueError("accepted MTP draft IDs disagree with the real output transcript")
                    # Native update computes rejection from the raw count, then
                    # appends one token at a time and stops at max_tokens/model
                    # length. Slots beyond that stop are never appended or hashed.
                    runner_output.sampled_token_ids[index] = real_tokens + [
                        self.scheduler_config.dummy_token_id
                    ] * (raw_generated_tokens[request_id] - visible)
                elif raw_generated_tokens[request_id]:
                    # Zero-output prompt completion still needs a terminal sample.
                    # Disable hashing before its dummy identity reaches native state.
                    self.scheduler.requests[request_id].get_hash_new_full_blocks = None
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
        draft_rows = None
        if mtp_drafter_executed:
            if self.scheduler_config.prefix_cache_mode == "token_ids":
                draft_rows = _validated_mtp_draft_rows(
                    self.scheduler_config.mtp_draft_token_ids_by_step,
                    step.schedule.step_id, tuple(native_output.num_scheduled_tokens),
                    self.scheduler_config.mtp_num_speculative_tokens,
                    self.vllm_config.model_config.get_vocab_size())
            else:
                draft_rows = [[self.scheduler_config.dummy_token_id]
                    * scheduled_by_id[request_id].mtp_proposer_tokens
                    for request_id in native_output.num_scheduled_tokens]
        update_started_ns = (
            time.perf_counter_ns()
            if self._service_measurement_samples is not None else None)
        native_results = self.scheduler.update_from_output(
            native_output, runner_output)
        if update_started_ns is not None:
            self._service_measurement_samples["update_ns"].append(
                time.perf_counter_ns() - update_started_ns)
        if draft_rows is not None:
            from vllm.v1.outputs import DraftTokenIds
            self.scheduler.update_draft_token_ids(DraftTokenIds(
                req_ids=list(native_output.num_scheduled_tokens),
                draft_token_ids=draft_rows,
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

    def close(self) -> None:
        if self.has_work():
            raise RuntimeError("cannot close a prepared vLLM bridge with unfinished work")
        self.assert_mtp_contract_exhausted()

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
        if self._dsv4_model is not None:
            # Lifecycle-exact reporting charge: streams across the whole
            # range, windowed caches only their in-window intersection
            # (review-06 R1).
            exact_bytes = self._dsv4_model.dsv4_cache_event_bytes(start, end)
            event_bytes = (
                -(-exact_bytes.numerator // exact_bytes.denominator))
        else:
            event_bytes = (end - start) * self.config.kv_bytes_per_token
        return KVCacheEvent(
            request_id=request_id,
            action=action,
            tier="hbm",
            token_start=start,
            token_end=end,
            bytes=event_bytes,
        )

    def _capacity_events(
        self,
        request_id: str,
        action: KVCacheAction,
        before_blocks: tuple[tuple[int, ...], ...],
        after_blocks: tuple[tuple[int, ...], ...],
    ) -> list[KVCacheEvent]:
        """Emit capacity events for changed pages in each cache group.

        Each group's block range is expressed in that group's own block
        tokens; bytes charge the group's real page bytes, so windowed
        DeepSeek-V4 caches report their bounded per-request footprint. Native
        mixed groups charge physical pool rows and exclude null placeholders;
        separate contiguous slot runs preserve sliding-window release ranges.
        """

        if action not in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE):
            raise ValueError(
                "capacity events support only allocate/release actions")
        events = []
        if self._prepared_v4:
            null_id = self.scheduler.kv_cache_manager.block_pool.null_block.block_id
            for group_index, (before, after) in enumerate(zip(before_blocks, after_blocks, strict=True)):
                target, other = (after, before) if action is KVCacheAction.ALLOCATE else (before, after)
                retained = set(other)
                changed = [index for index, block_id in enumerate(target) if block_id != null_id and block_id not in retained]
                block_tokens, pool_row_bytes = self._kv_group_geometry[group_index]
                runs = []
                for index in changed:
                    if runs and runs[-1][1] == index:
                        runs[-1][1] = index + 1
                    else:
                        runs.append([index, index + 1])
                for start, end in runs:
                    events.append(KVCacheEvent(request_id=request_id, action=action, tier="hbm",
                        token_start=start * block_tokens, token_end=end * block_tokens,
                        bytes=(end - start) * pool_row_bytes))
            return events
        for group_index, (before, after) in enumerate(
                zip(before_blocks, after_blocks, strict=True)):
            if action is KVCacheAction.ALLOCATE:
                start_block, end_block = len(before), len(after)
            else:
                start_block, end_block = 0, len(before)
            if end_block <= start_block:
                continue
            block_tokens, page_bytes = self._kv_group_geometry[group_index]
            events.append(KVCacheEvent(
                request_id=request_id,
                action=action,
                tier="hbm",
                token_start=start_block * block_tokens,
                token_end=end_block * block_tokens,
                bytes=(end_block - start_block) * page_bytes,
            ))
        return events

    def _block_ids(self, request_id: str) -> tuple[tuple[int, ...], ...]:
        block_groups = self.scheduler.kv_cache_manager.get_blocks(
            request_id).get_block_ids()
        if len(block_groups) != len(self._kv_group_geometry):
            raise RuntimeError(
                "vLLM KV cache group count changed after construction: "
                f"{len(block_groups)} vs {len(self._kv_group_geometry)}")
        normalized = tuple(
            tuple(_vllm_integer(block_id, "KV block ID")
                  for block_id in group)
            for group in block_groups
        )
        null_block_id = _vllm_integer(
            self.scheduler.kv_cache_manager.block_pool.null_block.block_id,
            "KV null block ID")
        for group_index, group in enumerate(normalized):
            live = [block_id for block_id in group
                    if block_id != null_block_id]
            if len(live) != len(set(live)):
                raise RuntimeError(
                    "vLLM allocated duplicate KV block IDs within a group: "
                    f"request={request_id} group={group_index} ids={group}")
        return normalized


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
    if scheduler_config.scheduler_version != "0.11.0":
        raise NotImplementedError("from_model remains the pinned 0.11 path; use the explicit prepared 0.30 bridge")
    from src.serving.workload.model_config import load_model_config
    scope_model = load_model_config(scheduler_config.model)
    if scheduler_config.prefix_cache_mode == "token_ids" and scope_model.model_type == "deepseek_v4":
        raise NotImplementedError("token prefix cache requires a supported single KV cache group; V4 mixed cache is unsupported")
    if scope_model.config_sha256 == KIMI_K3_CONFIG_SHA256:
        raise NotImplementedError("official Kimi-K3 requires hybrid KDA/NoPE MLA vLLM 0.30 capacity and Scheduler contracts")
    if scope_model.config_sha256 == KIMI_K2_7_CODE_CONFIG_SHA256:
        raise NotImplementedError(
            "official Kimi-K2.7-Code has no bound vLLM 0.30 normal Scheduler adapter; "
            "the existing from_model Scheduler is pinned to 0.11")
    if scope_model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S:
        raise NotImplementedError(
            "official GLM block-FP8 requires the explicit vLLM 0.30 adapter; "
            "the existing production Scheduler is pinned to 0.11")
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
    dense_colocated_block128 = (
        config.kv_block_size_tokens == 128
        and config.role == "colocated"
        and scheduler_config.tensor_parallel_size == 1
        and scheduler_config.pipeline_parallel_size == 1
        and not scheduler_config.enable_expert_parallel
        and scheduler_config.mtp_num_speculative_tokens == 0
        and scope_model.model_type == "qwen3"
        and scope_model.attention_layout == "standard_gqa"
        and scope_model.num_experts == 0
        and scope_model.compute_dtype == scope_model.weight_dtype == "bf16"
    )
    # The CPU Scheduler and FullAttentionSpec receive the requested geometry
    # directly; no GPU worker or version-specific attention backend is selected.
    # Keep all other nonbaseline block geometries outside this validated slice.
    if config.kv_block_size_tokens != 16 and not dense_colocated_block128:
        raise NotImplementedError(
            "the pinned CPU-only vLLM Scheduler requires a 16-token KV block "
            "size outside the Qwen3 BF16 colocated TP1 PP1 MTP0 block128 slice")
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
            SlidingWindowSpec,
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
    dsv4_capacity_context = None
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
            enable_prefix_caching=(scheduler_config.prefix_cache_mode == "token_ids"),
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
        elif model.attention_layout == "dsv4_sparse_mla":
            dsv4_parallel_scope = (
                (model.config_sha256 == DEEPSEEK_V4_FLASH_CONFIG_SHA256
                 and (scheduler_config.tensor_parallel_size,
                      scheduler_config.pipeline_parallel_size,
                      scheduler_config.enable_expert_parallel,
                      config.role) == (4, 1, False, "colocated"))
                or (model.config_sha256 == DEEPSEEK_V4_PRO_CONFIG_SHA256
                    and (scheduler_config.tensor_parallel_size,
                         scheduler_config.pipeline_parallel_size,
                         scheduler_config.enable_expert_parallel,
                         config.role) == (8, 1, False, "colocated"))
            )
            if (model.model_type != "deepseek_v4"
                    or model.architecture != "DeepseekV4ForCausalLM"
                    or model.compute_dtype != "bf16"
                    or model.weight_dtype != "fp8"
                    or scheduler_config.dtype != "bfloat16"
                    or not dsv4_parallel_scope
                    or scheduler_config.mtp_num_speculative_tokens > 0):
                raise NotImplementedError(
                    "the validated scheduler-only DeepSeek-V4 sparse-MLA "
                    "slice supports exactly DeepSeek-V4-Flash FP8 with "
                    "TP4/PP1 colocated or DeepSeek-V4-Pro FP8 with TP8/PP1 "
                    "colocated on H200, with MTP disabled")
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
            enable_prefix_caching=(scheduler_config.prefix_cache_mode == "token_ids"),
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
        if model.attention_layout == "dsv4_sparse_mla":
            # DeepSeek-V4 is PP=1-only (rank-specific compress-ratio layer
            # placement), so the single stage is trivially representative.
            representative_stage = 0
        else:
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
        elif model.attention_layout == "dsv4_sparse_mla":
            # DeepSeek-V4 decomposes into per-lifecycle Scheduler cache
            # groups (ModelConfig.dsv4_scheduler_cache_groups): windowed
            # caches (SWA window, FP32 compressor partial states) become
            # SlidingWindow groups whose per-request footprint caps at the
            # upstream window, and unbounded streams (compressed KV,
            # indexer K) become FullAttention groups with the upstream
            # 576-byte-aligned page geometry.  With prefix caching disabled
            # the pinned 0.11.0 coordinator imposes no group-type
            # constraints, so each group gets its real manager lifecycle.
            dsv4_groups = model.dsv4_scheduler_cache_groups(
                config.kv_block_size_tokens)
            dsv4_capacity_context = (dsv4_groups, model)
            num_kv_heads = 1
            head_size = 0
            kv_spec_layout = "dsv4_lifecycle"
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
    elif kv_spec_layout == "dsv4_lifecycle":
        cache_spec = None
    else:
        raise AssertionError(
            f"unhandled Scheduler KV spec layout: {kv_spec_layout!r}")
    if kv_spec_layout == "dsv4_lifecycle":
        # One Scheduler group per cache lifecycle class; virtual layer names
        # carry the component kind so each physical layer's caches join the
        # groups of their real lifecycle (review-06 R1).
        dsv4_kv_group_specs = []
        dsv4_kv_tensors = []
        for group in dsv4_groups:
            if group.lifecycle == "sliding_window":
                # SlidingWindowSpec pages charge the K+V pair
                # (2 * block * head * dtype); the FP32 state pages divide
                # exactly, the SWA row is ceiled to an even 586 bytes in the
                # ModelConfig group table.
                if group.page_bytes % (2 * group.block_tokens):
                    raise RuntimeError(
                        "DeepSeek-V4 sliding cache group page must cover "
                        f"whole K+V token bytes: kind={group.kind!r}, "
                        f"page={group.page_bytes}, "
                        f"block_tokens={group.block_tokens}")
                group_spec = SlidingWindowSpec(
                    block_size=group.block_tokens,
                    num_kv_heads=1,
                    head_size=group.page_bytes // (2 * group.block_tokens),
                    dtype=torch.uint8,
                    sliding_window=group.window_tokens,
                )
            elif group.lifecycle == "stream":
                # MLAAttentionSpec pages charge the single compressed
                # latent row (block * head * dtype), matching the upstream
                # compressed-KV/indexer-K storage.
                if group.page_bytes % group.block_tokens:
                    raise RuntimeError(
                        "DeepSeek-V4 stream cache group page must cover "
                        f"whole token bytes: kind={group.kind!r}, "
                        f"page={group.page_bytes}, "
                        f"block_tokens={group.block_tokens}")
                group_spec = MLAAttentionSpec(
                    block_size=group.block_tokens,
                    num_kv_heads=1,
                    head_size=group.page_bytes // group.block_tokens,
                    dtype=torch.uint8,
                    cache_dtype_str=None,
                )
            else:
                raise AssertionError(
                    f"unhandled DeepSeek-V4 cache lifecycle: "
                    f"{group.lifecycle!r}")
            if group_spec.page_size_bytes != group.page_bytes:
                raise RuntimeError(
                    "DeepSeek-V4 Scheduler spec page disagrees with the "
                    f"ModelConfig cache group: kind={group.kind!r}, "
                    f"spec={group_spec.page_size_bytes}, "
                    f"model={group.page_bytes}")
            group_layer_names = [
                f"model.layers.{index}.self_attn.attn:{group.kind}"
                for index in group.layer_indices
            ]
            dsv4_kv_group_specs.append(KVCacheGroupSpec(
                layer_names=group_layer_names, kv_cache_spec=group_spec))
            dsv4_kv_tensors.extend(
                KVCacheTensor(
                    size=config.num_kv_blocks * group.page_bytes,
                    shared_by=[layer_name],
                )
                for layer_name in group_layer_names
            )
        kv_cache_config = KVCacheConfig(
            num_blocks=config.num_kv_blocks,
            kv_cache_tensors=dsv4_kv_tensors,
            kv_cache_groups=dsv4_kv_group_specs,
        )
        layer_names = [
            layer_name
            for group in dsv4_kv_group_specs
            for layer_name in group.layer_names
        ]
        page_bytes = None
    elif model.model_type in ("kimi_k2", "glm_moe_dsa", "deepseek_v3",
                            "deepseek_v4"):
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
    if kv_spec_layout != "dsv4_lifecycle":
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
    elif kv_spec_layout == "dsv4_lifecycle":
        tensor_page_bytes = None
    else:
        raise AssertionError(
            f"unhandled Scheduler tensor layout: {kv_spec_layout!r}")
    if kv_spec_layout != "dsv4_lifecycle":
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
                KVCacheGroupSpec(
                    layer_names=layer_names, kv_cache_spec=cache_spec)
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
    if kv_spec_layout != "dsv4_lifecycle":
        if page_bytes % config.kv_block_size_tokens:
            raise RuntimeError(
                "vLLM Scheduler KV page bytes must divide by block tokens")
    total_tensor_bytes = sum(
        tensor.size for tensor in kv_cache_config.kv_cache_tensors)
    capacity_denominator = (
        config.num_kv_blocks * config.kv_block_size_tokens)
    if model.attention_layout == "dsv4_sparse_mla":
        # The lifecycle groups no longer have a uniform per-token byte rate:
        # windowed caches cap per request at their window, streams charge
        # 576-aligned pages.  The constructed tensors are validated
        # component-by-component against the ModelConfig group table above;
        # BackendConfig keeps the conservative ceiled full-rate per-token
        # total as the cross-lane reporting charge (evaluation computes it
        # from the same ModelConfig Fraction).
        model_kv_bytes_per_token = (
            model.kv_bytes_per_token_per_parallel_rank(
                scheduler_config.tensor_parallel_size,
                scheduler_config.pipeline_parallel_size))
        model_kv_bytes_per_token_ceiled = (
            -(-model_kv_bytes_per_token.numerator
              // model_kv_bytes_per_token.denominator))
        if config.kv_bytes_per_token != model_kv_bytes_per_token_ceiled:
            raise RuntimeError(
                "BackendConfig KV bytes disagree with the ceiled DeepSeek-V4 "
                "per-rank attention-cache capacity: "
                f"backend={config.kv_bytes_per_token}, "
                f"model={model_kv_bytes_per_token_ceiled}")
    else:
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
    return scheduler, vllm_config, dsv4_capacity_context


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
    if cache.enable_prefix_caching != (requested.prefix_cache_mode == "token_ids"):
        raise RuntimeError("vLLM effective prefix caching disagrees with requested mode")
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
    scheduler_version: str = "0.11.0",
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
    block_hasher = None
    request.validate_token_identity(vocab_size, require_complete=vllm_config.cache_config.enable_prefix_caching)
    if vllm_config.cache_config.enable_prefix_caching:
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher
        from vllm.utils import get_hash_fn_by_name
        hash_fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
        block_hasher = get_request_block_hasher(vllm_config.cache_config.block_size, hash_fn)
    version_fields = {}
    if scheduler_version == "0.11.0":
        version_fields["eos_token_id"] = None
    elif scheduler_version != "0.30.0":
        raise ValueError("unsupported vLLM Request API version")
    return Request(
        request_id=request.request_id,
        prompt_token_ids=(list(request.prompt_token_ids) if request.prompt_token_ids is not None
                          else _LengthOnlyTokenIds(dummy_token_id, request.input_tokens)),
        sampling_params=SamplingParams(
            max_tokens=output_limit),
        pooling_params=None,
        arrival_time=request.arrival_time_s,
        block_hasher=block_hasher,
        **version_fields,
    )


def _resumed_native_request_ids(cached: Any, request_ids: tuple[str, ...],
                                scheduler_version: str) -> tuple[str, ...]:
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("vLLM cached request IDs contain duplicates")
    if scheduler_version == "0.30.0":
        resumed = cached.resumed_req_ids
        if not isinstance(resumed, set) or any(not isinstance(value, str) for value in resumed):
            raise TypeError("vLLM 0.30 resumed_req_ids must be a set of request IDs")
        if not resumed.issubset(request_ids):
            raise ValueError("vLLM 0.30 resumed an uncached request")
        return tuple(request_id for request_id in request_ids if request_id in resumed)
    if scheduler_version == "0.11.0":
        flags = tuple(cached.resumed_from_preemption)
        if len(request_ids) != len(flags):
            raise RuntimeError("vLLM returned inconsistent resumed-from-preemption metadata")
        if any(not isinstance(flag, bool) for flag in flags):
            raise TypeError("vLLM resumed-from-preemption flags must be boolean")
        return tuple(request_id for request_id, resumed in zip(request_ids, flags, strict=True) if resumed)
    raise ValueError("unsupported vLLM cached request API version")


def _verify_content_hash_state_disabled(native_request: Any,
                                        scheduler_version: str = "0.11.0") -> None:
    """Fail if a synthetic length-only request acquires content identity."""

    if native_request.block_hashes != []:
        raise RuntimeError(
            "length-only vLLM request unexpectedly contains block hashes")
    if scheduler_version == "0.11.0":
        hasher = native_request.get_hash_new_full_blocks
    elif scheduler_version == "0.30.0":
        hasher = native_request._block_hasher
    else:
        raise ValueError("unsupported vLLM request hashing API version")
    if hasher is not None:
        raise RuntimeError(
            "length-only vLLM request unexpectedly enables content hashing")


def _validated_mtp_draft_rows(frames, step_id, request_ids, depth, vocab_size):
    """Validate the actual executed proposer batch immediately before feedback."""
    if frames is None or step_id not in frames:
        raise ValueError(f"executed MTP proposer requires real draft IDs for step {step_id!r}")
    frame = frames[step_id]
    if not isinstance(frame, Mapping) or set(frame) != set(request_ids):
        raise ValueError("MTP draft request IDs must exactly match the executed proposer batch")
    if any(not isinstance(tokens, tuple) or len(tokens) != depth
            or any(type(token) is not int or token < 0 for token in tokens)
            for tokens in frame.values()):
        raise ValueError("MTP draft IDs must be nonnegative integer tuples covering proposer depth")
    if any(token >= vocab_size for tokens in frame.values() for token in tokens):
        raise ValueError("MTP draft IDs must be within the model vocabulary")
    return [list(frame[request_id]) for request_id in request_ids]


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
