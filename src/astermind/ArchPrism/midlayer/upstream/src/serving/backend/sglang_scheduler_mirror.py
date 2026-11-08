"""Source-mirrored SGLang Scheduler admission for fixed serving recipes.

This backend reproduces the batch-selection semantics of the SGLang
``Scheduler`` for the recipe that InferenceX runs on DeepSeek-R1:
``disable_radix_cache=True`` (``ChunkCache``), ``enable_mixed_chunk=False``,
``page_size=1``, FCFS, ``--ignore-eos`` requests, chunked prefill and an
optional EAGLE proposer.  Every rule below cites the pinned upstream source
(``python/sglang/srt`` of v0.5.19, commit 0bcd822377da7b5718e674eaf9c870d349424dd1):

* ``managers/scheduler.py:get_next_batch_to_run`` -- a new prefill batch has
  priority; otherwise the running batch decodes.  Prefill and decode are never
  mixed because ``enable_mixed_chunk`` is False (``running_bs if
  self.is_mixed_chunk else 0`` passed to ``PrefillAdder``).
* ``managers/scheduler.py:_get_new_batch_prefill_raw`` -- ``batch_is_full``
  short-circuit, the chunked request first, then FCFS waiting requests until
  ``AddReqResult`` stops the loop; ``NO_TOKEN`` marks the batch full.
* ``managers/schedule_policy.py:PrefillAdder`` -- ``rem_total_tokens``,
  ``cur_rem_tokens``, ``add_chunked_req``, ``add_one_req_ignore_eos`` (taken
  because ``ChunkCache.disable`` is True and every request sets ignore_eos),
  ``_update_prefill_budget`` and ``budget_state``.
* ``scheduler_components/new_token_ratio_tracker.py`` and ``environ.py`` --
  the running-request reservation ratio (init 0.7, floor factor 0.14, 600
  decay steps), decayed once per decode batch in ``update_running_batch``.

KV capacity (``max_total_num_tokens``) is a serving configuration fact that
the caller must supply with its provenance.  Retraction is not mirrored: the
ignore-eos admission reserves every admitted request's full output, and a
decode that would still exceed the pool fails closed instead of inventing a
retraction trajectory.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

from src.serving.backend.mtp_acceptance_assumption import (
    MTPAcceptanceAssumption,
)
from src.serving.request_gen.request_stream import RequestSpec
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduledRequest,
    ScheduleStep,
)

SGLANG_SOURCE_VERSIONS = ("v0.5.12", "v0.5.19")

# environ.py:559-561 and schedule_policy.py:76-78,96 (identical in v0.5.12).
_INIT_NEW_TOKEN_RATIO = 0.7
_MIN_NEW_TOKEN_RATIO_FACTOR = 0.14
_NEW_TOKEN_RATIO_DECAY_STEPS = 600
_CLIP_MAX_NEW_TOKENS = 4096
_IGNORE_EOS_RESERVE_TOKENS = 1

_CONTINUE = "continue"
_NO_TOKEN = "no_token"
_OTHER = "other"


@dataclass(frozen=True)
class SGLangSchedulerMirrorConfig:
    source_version: str
    max_total_num_tokens: int
    max_total_num_tokens_source: str
    chunked_prefill_size: int
    max_prefill_tokens: int
    max_running_requests: int
    schedule_conservativeness: float
    mtp_num_speculative_tokens: int
    mtp_acceptance_assumption: Optional[MTPAcceptanceAssumption]
    mtp_acceptance_source_id: str

    def __post_init__(self) -> None:
        if self.source_version not in SGLANG_SOURCE_VERSIONS:
            raise ValueError(
                f"unsupported SGLang source version: {self.source_version!r}")
        for name in ("max_total_num_tokens", "chunked_prefill_size",
                     "max_prefill_tokens", "max_running_requests"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"SGLang {name} must be a positive integer")
        if (not isinstance(self.max_total_num_tokens_source, str)
                or not self.max_total_num_tokens_source.strip()):
            raise ValueError(
                "SGLang max_total_num_tokens requires a provenance string")
        if not self.schedule_conservativeness > 0:
            raise ValueError("schedule_conservativeness must be positive")
        if self.mtp_num_speculative_tokens == 0:
            if (self.mtp_acceptance_assumption is not None
                    or self.mtp_acceptance_source_id != "disabled"):
                raise ValueError(
                    "disabled SGLang EAGLE must not carry an acceptance source")
        elif self.mtp_num_speculative_tokens > 0:
            if self.mtp_acceptance_assumption is None:
                raise ValueError(
                    "SGLang EAGLE requires an explicit acceptance assumption")
            if (self.mtp_acceptance_assumption.proposal_depth
                    != self.mtp_num_speculative_tokens):
                raise ValueError(
                    "SGLang EAGLE acceptance depth disagrees with the proposer")
        else:
            raise ValueError("speculative token count cannot be negative")

    def runtime_manifest(self) -> dict[str, object]:
        return {
            "framework": "sglang",
            "mode": "source_mirrored_scheduler",
            "source_version": self.source_version,
            "radix_cache": "disabled",
            "mixed_chunk": False,
            "page_size": 1,
            "schedule_policy": "fcfs",
            "ignore_eos": True,
            "retraction": "unsupported_fail_closed",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_total_num_tokens_source": self.max_total_num_tokens_source,
            "chunked_prefill_size": self.chunked_prefill_size,
            "max_prefill_tokens": self.max_prefill_tokens,
            "max_running_requests": self.max_running_requests,
            "schedule_conservativeness": self.schedule_conservativeness,
            "speculative_algorithm": (
                "EAGLE" if self.mtp_num_speculative_tokens else "disabled"),
            "mtp_num_speculative_tokens": self.mtp_num_speculative_tokens,
            "mtp_acceptance_source_id": self.mtp_acceptance_source_id,
        }


class _Request:
    __slots__ = ("spec", "prompt_computed", "output_produced",
                 "prompt_token_ids", "output_token_ids")

    def __init__(self, spec: RequestSpec, *,
                 prompt_token_ids: tuple[int, ...] | None,
                 output_token_ids: tuple[int, ...] | None) -> None:
        self.spec = spec
        self.prompt_computed = 0
        self.output_produced = 0
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = output_token_ids

    @property
    def kv_tokens(self) -> int:
        return self.prompt_computed + self.output_produced


class _PrefillAdder:
    """schedule_policy.py:PrefillAdder restricted to the mirrored recipe."""

    def __init__(self, available: int, running: list[_Request],
                 new_token_ratio: float, rem_input_tokens: int,
                 rem_chunk_tokens: int) -> None:
        self.available = available
        self.running = running
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        self.rem_total_token_offset = sum(
            min(r.spec.output_tokens - r.output_produced, _CLIP_MAX_NEW_TOKENS)
            * new_token_ratio
            for r in running)
        self.cur_rem_token_offset = 0
        self.can_run: list[tuple[_Request, int]] = []
        self.new_chunked: Optional[_Request] = None
        self.req_states: Optional[list[tuple[int, int]]] = None

    @property
    def rem_total_tokens(self) -> float:
        return self.available - self.rem_total_token_offset

    @property
    def cur_rem_tokens(self) -> float:
        return self.available - self.cur_rem_token_offset

    def budget_state(self) -> str:
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return _NO_TOKEN
        if self.rem_input_tokens <= 0:
            return _OTHER
        if self.rem_chunk_tokens <= 0:
            return _OTHER
        return _CONTINUE

    def _update_prefill_budget(self, extend: int, max_new: int) -> None:
        self.rem_total_token_offset += extend + max_new + 1
        self.cur_rem_token_offset += extend + 1
        self.rem_input_tokens -= extend
        self.rem_chunk_tokens -= extend

    def add_chunked_req(self, request: _Request) -> Optional[_Request]:
        rem = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
        if rem <= 0:
            rem = self.rem_chunk_tokens
        candidate = request.spec.input_tokens - request.prompt_computed
        truncated = candidate > rem
        extend = min(candidate, rem)
        self.can_run.append((request, extend))
        self._update_prefill_budget(
            extend,
            0 if truncated else min(
                request.spec.output_tokens, _CLIP_MAX_NEW_TOKENS),
        )
        return request if truncated else None

    def _req_state(self, request: _Request) -> Optional[tuple[int, int]]:
        # ignore_eos requests use ratio 1.0 (schedule_policy.py add_req_state).
        tokens_left = request.spec.output_tokens - request.output_produced
        if tokens_left <= 0:
            return None
        return (tokens_left, request.spec.input_tokens + request.output_produced)

    def add_one_req_ignore_eos(self, request: _Request) -> str:
        extend = request.spec.input_tokens - request.prompt_computed
        if extend > min(self.cur_rem_tokens, self.rem_total_tokens):
            return _NO_TOKEN
        if self.req_states is None:
            states = [self._req_state(request)]
            states.extend(self._req_state(r) for r in self.running)
            states.extend(self._req_state(r) for r, _ in self.can_run)
            self.req_states = sorted(
                (state for state in states if state is not None),
                key=lambda state: state[0])
        else:
            state = self._req_state(request)
            if state is not None:
                keys = [s[0] for s in self.req_states]
                # Upstream inserts before the first entry with left >= state.
                index = bisect.bisect_left(keys, state[0])
                if index == len(keys) and keys:
                    index = len(keys) - 1
                self.req_states.insert(index, state)
        cur_rem_tokens = self.cur_rem_tokens - extend
        tokens_freed = 0
        count = len(self.req_states)
        for index, (tokens_left, tokens_occupied) in enumerate(self.req_states):
            bs = count - index
            min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
            if min_free_tokens <= _IGNORE_EOS_RESERVE_TOKENS * bs:
                return _NO_TOKEN
            tokens_freed += tokens_occupied
        if extend <= self.rem_chunk_tokens:
            self.can_run.append((request, extend))
            self._update_prefill_budget(
                extend,
                min(request.spec.output_tokens, _CLIP_MAX_NEW_TOKENS))
        else:
            if self.rem_chunk_tokens <= 0:
                return _OTHER
            trunc = self.rem_chunk_tokens
            self.can_run.append((request, trunc))
            self.new_chunked = request
            self._update_prefill_budget(trunc, 0)
        return self.budget_state()


class SGLangSchedulerMirrorBackend:
    """Deterministic SGLang Scheduler mirror; see the module docstring."""

    def __init__(self, config: BackendConfig,
                 sglang_config: SGLangSchedulerMirrorConfig, *,
                 request_token_transcripts: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
                 token_transcript_source_id: str | None = None,
                 model_source_inputs_by_step: Mapping[str, object] | None = None) -> None:
        if config.role != "colocated":
            raise NotImplementedError(
                "the SGLang Scheduler mirror supports only colocated pools")
        if config.max_batched_tokens != sglang_config.chunked_prefill_size:
            raise ValueError(
                "SGLang chunked_prefill_size must equal max_batched_tokens")
        if (request_token_transcripts is None) != (token_transcript_source_id is None):
            raise ValueError("SGLang request transcripts require their explicit source identity")
        if request_token_transcripts is not None:
            if (not isinstance(request_token_transcripts, Mapping) or not request_token_transcripts
                    or not isinstance(token_transcript_source_id, str) or not token_transcript_source_id.strip()):
                raise ValueError("SGLang request transcripts require a nonempty mapping and source identity")
            for request_id, pair in request_token_transcripts.items():
                if (not isinstance(request_id, str) or not request_id or not isinstance(pair, tuple)
                        or len(pair) != 2 or any(not isinstance(values, tuple) for values in pair)
                        or any(type(token) is not int or not 0 <= token < 2**31 for values in pair for token in values)):
                    raise ValueError("SGLang request transcript must contain a prompt/output pair of INT32 tuples")
            if sglang_config.mtp_num_speculative_tokens and model_source_inputs_by_step is None:
                raise NotImplementedError("SGLang EAGLE token export requires explicit draft/tree state")
        if model_source_inputs_by_step is not None:
            from src.serving.workload.traced_sglang import SGLangEagleChainState, SGLangPrefillInputs, SGLangGraphInputs
            if (not isinstance(model_source_inputs_by_step, Mapping) or not model_source_inputs_by_step
                    or request_token_transcripts is None or sglang_config.source_version != "v0.5.19"
                    or sglang_config.mtp_num_speculative_tokens not in (0, 1, 2)):
                raise ValueError("SGLang typed model source requires explicit transcripts and its0.5.19 depth0/1/2 domain")
            for step_id, state in model_source_inputs_by_step.items():
                if (not isinstance(step_id, str) or not step_id.startswith(config.pool_name + ":sglang:")
                        or (sglang_config.mtp_num_speculative_tokens > 0 and (
                            not isinstance(state, SGLangEagleChainState)
                            or state.proposal_depth != sglang_config.mtp_num_speculative_tokens))
                        or (sglang_config.mtp_num_speculative_tokens == 0 and state is not None
                            and not isinstance(state, (SGLangPrefillInputs, SGLangGraphInputs)))):
                    raise ValueError("SGLang model-source inputs require exact step IDs and typed states")
        self._model_source_inputs_by_step = (None if model_source_inputs_by_step is None else
            MappingProxyType(dict(model_source_inputs_by_step)))
        self._last_eagle_inputs_by_request = {}
        self._request_token_transcripts = (None if request_token_transcripts is None else
            MappingProxyType(dict(request_token_transcripts)))
        self._token_transcript_source_id = token_transcript_source_id
        self.config = config
        self.sglang_config = sglang_config
        init = min(
            _INIT_NEW_TOKEN_RATIO * sglang_config.schedule_conservativeness,
            1.0)
        self._ratio_min = min(init * _MIN_NEW_TOKEN_RATIO_FACTOR, 1.0)
        self._ratio_decay = (init - self._ratio_min) / _NEW_TOKEN_RATIO_DECAY_STEPS
        self._new_token_ratio = init
        self._waiting: list[_Request] = []
        self._running: list[_Request] = []
        self._chunked: Optional[_Request] = None
        self._batch_is_full = False
        self._pending: Optional[BackendStep] = None
        self._pending_prefill: Optional[tuple[_Request, ...]] = None
        self._step = 0
        self._ids: set[str] = set()
        self._explicit_token_ids: bool | None = None

    @property
    def runtime_info(self) -> dict[str, object]:
        return {**self.sglang_config.runtime_manifest(),
            "request_token_transcript_source_id": self._token_transcript_source_id,
            "model_source_input_source_ids": (None if self._model_source_inputs_by_step is None else
                tuple(sorted({state.source_id for state in self._model_source_inputs_by_step.values() if state is not None})))}

    def add_request(self, request: RequestSpec, *,
                    prompt_token_ids: tuple[int, ...] | None = None,
                    output_token_ids: tuple[int, ...] | None = None) -> None:
        """Admit lengths, optionally with an explicit external token transcript.

        The mirror does not execute a model or synthesize sampled token values.
        A value-aware caller supplies the prompt and output transcript together;
        ordinary decode consumes the preceding declared output token. Length-only
        callers retain the existing scheduling behavior and export no token IDs.
        EAGLE additionally needs draft token/tree state, which this interface does
        not pretend can be recovered from an accepted-token count.
        """
        if not isinstance(request, RequestSpec):
            raise TypeError("SGLang admission requires RequestSpec")
        if request.request_id in self._ids:
            raise ValueError(f"duplicate request id: {request.request_id!r}")
        if request.prefix_hit_tokens != 0:
            raise NotImplementedError(
                "the SGLang mirror runs with the radix cache disabled")
        if request.output_tokens < 1:
            raise NotImplementedError(
                "the SGLang mirror requires at least one output token")
        if self._request_token_transcripts is not None:
            if prompt_token_ids is not None or output_token_ids is not None:
                raise ValueError("SGLang request transcripts must have exactly one input owner")
            if request.request_id not in self._request_token_transcripts:
                raise ValueError("SGLang request is absent from the explicitly bound token transcripts")
            prompt_token_ids, output_token_ids = self._request_token_transcripts[request.request_id]
        if (prompt_token_ids is None) != (output_token_ids is None):
            raise ValueError("SGLang explicit tokens require both prompt and output transcripts")
        explicit = prompt_token_ids is not None
        if explicit:
            if self.sglang_config.mtp_num_speculative_tokens and self._model_source_inputs_by_step is None:
                raise NotImplementedError("SGLang EAGLE token export requires explicit draft/tree state")
            for name, values, count in (
                    ("prompt_token_ids", prompt_token_ids, request.input_tokens),
                    ("output_token_ids", output_token_ids, request.output_tokens)):
                if (not isinstance(values, tuple) or len(values) != count
                        or any(type(token) is not int or not 0 <= token < 2**31
                               for token in values)):
                    raise ValueError(f"SGLang {name} must be an INT32 token tuple covering its request length")
        if self._explicit_token_ids is not None and self._explicit_token_ids != explicit:
            raise ValueError("SGLang cannot mix explicit-token and length-only requests in one backend")
        self._explicit_token_ids = explicit
        self._ids.add(request.request_id)
        self._waiting.append(_Request(request, prompt_token_ids=prompt_token_ids,
                                      output_token_ids=output_token_ids))

    def has_work(self) -> bool:
        return bool(self._waiting or self._running or self._chunked
                    or self._pending is not None)

    def close(self) -> None:
        return None

    def _available_tokens(self) -> int:
        used = sum(r.kv_tokens for r in self._running)
        if self._chunked is not None:
            used += self._chunked.kv_tokens
        return self.sglang_config.max_total_num_tokens - used

    def _new_prefill(self) -> Optional[list[tuple[_Request, int]]]:
        if ((self._batch_is_full or not self._waiting)
                and self._chunked is None):
            return None
        running_bs = len(self._running)
        live = running_bs + (1 if self._chunked is not None else 0)
        allocatable = min(
            self.sglang_config.max_running_requests - running_bs,
            self.sglang_config.max_running_requests - live)
        if allocatable <= 0 and self._chunked is None:
            self._batch_is_full = True
            return None
        adder = _PrefillAdder(
            self._available_tokens(),
            self._running,
            self._new_token_ratio,
            self.sglang_config.max_prefill_tokens,
            self.sglang_config.chunked_prefill_size,
        )
        if self._chunked is not None:
            self._chunked = adder.add_chunked_req(self._chunked)
        for request in self._waiting:
            if len(adder.can_run) >= allocatable:
                self._batch_is_full = True
                break
            result = adder.add_one_req_ignore_eos(request)
            if result != _CONTINUE:
                if result == _NO_TOKEN:
                    self._batch_is_full = True
                break
        if not adder.can_run:
            return None
        admitted = {id(r) for r, _ in adder.can_run}
        self._waiting = [r for r in self._waiting if id(r) not in admitted]
        if adder.new_chunked is not None:
            if self._chunked is not None:
                raise AssertionError("two chunked SGLang requests")
            self._chunked = adder.new_chunked
        return adder.can_run

    def _mtp_fields(self, *, proposed: int, verified: int, accepted: int,
                    finished: bool) -> dict[str, object]:
        depth = self.sglang_config.mtp_num_speculative_tokens
        if depth == 0:
            return {}
        return {
            "mtp_proposed_tokens": proposed,
            "mtp_verified_tokens": verified,
            "mtp_accepted_tokens": accepted,
            "mtp_proposer_tokens": depth,
            "mtp_next_proposed_tokens": 0 if finished else depth,
            "mtp_proposer_mode": "padded",
            "mtp_acceptance_source_kind": "explicit_assumption",
            "mtp_acceptance_source_id": (
                self.sglang_config.mtp_acceptance_source_id),
        }

    def plan_step(self, virtual_time_s: float) -> Optional[BackendStep]:
        if self._pending is not None:
            raise RuntimeError("complete the SGLang step before planning another")
        step_id = f"{self.config.pool_name}:sglang:{self._step}"
        state = None
        if self._model_source_inputs_by_step is not None:
            if not self.has_work():
                return None
            if step_id not in self._model_source_inputs_by_step:
                raise ValueError("SGLang step is absent from the explicitly bound model-source inputs")
            state = self._model_source_inputs_by_step[step_id]
        scheduled = []
        generated = {}
        finished = []
        scheduled_token_ids = [] if self._explicit_token_ids else None
        prefill = self._new_prefill()
        if prefill is not None:
            for request, extend in prefill:
                if scheduled_token_ids is not None:
                    scheduled_token_ids.extend(request.prompt_token_ids[
                        request.prompt_computed:request.prompt_computed + extend])
                completes = (request.prompt_computed + extend
                             == request.spec.input_tokens)
                raw = 1 if completes else 0
                done = completes and request.spec.output_tokens <= 1
                if done:
                    finished.append(request.spec.request_id)
                generated[request.spec.request_id] = raw
                scheduled.append(ScheduledRequest(
                    request_id=request.spec.request_id,
                    phase="prefill",
                    input_tokens=extend,
                    output_tokens=0,
                    computed_tokens=request.prompt_computed,
                    raw_model_output_tokens=raw,
                    **self._mtp_fields(proposed=0, verified=0, accepted=0,
                                       finished=done),
                ))
            self._pending_prefill = tuple(r for r, _ in prefill)
        elif self._running:
            depth = self.sglang_config.mtp_num_speculative_tokens
            required = len(self._running) * (depth + 1)
            if self._available_tokens() < required:
                raise NotImplementedError(
                    "SGLang decode would exceed the KV pool; retraction is "
                    "not mirrored")
            self._new_token_ratio = max(
                self._new_token_ratio - self._ratio_decay, self._ratio_min)
            if scheduled_token_ids is not None and depth and (
                    state is None or state.phase != "verify"
                    or state.request_ids != tuple(value.spec.request_id for value in self._running)):
                raise ValueError("SGLang EAGLE verify requires its exact explicit chain state")
            for row, request in enumerate(self._running):
                if scheduled_token_ids is not None:
                    if request.output_produced < 1:
                        raise RuntimeError("SGLang decode token export lost its output prefix")
                    scheduled_token_ids.append(request.output_token_ids[request.output_produced - 1])
                    if depth:
                        scheduled_token_ids.extend(state.draft_token_ids[row * depth:(row + 1) * depth])
                if depth:
                    accepted = self.sglang_config.mtp_acceptance_assumption.accepted_count(
                        step_id=step_id,
                        request_id=request.spec.request_id,
                        proposed_tokens=depth,
                    )
                else:
                    accepted = 0
                raw = accepted + 1
                remaining = request.spec.output_tokens - request.output_produced
                done = raw >= remaining
                if done:
                    finished.append(request.spec.request_id)
                generated[request.spec.request_id] = min(raw, remaining)
                scheduled.append(ScheduledRequest(
                    request_id=request.spec.request_id,
                    phase="decode",
                    input_tokens=0,
                    output_tokens=depth + 1,
                    computed_tokens=(request.spec.input_tokens
                                     + request.output_produced - 1),
                    raw_model_output_tokens=raw,
                    **self._mtp_fields(proposed=depth, verified=depth,
                                       accepted=accepted, finished=done),
                ))
            self._pending_prefill = None
        else:
            if self._waiting or self._chunked is not None:
                raise RuntimeError(
                    "SGLang mirror cannot admit a waiting request into an "
                    "empty engine; the request exceeds the KV pool")
            return None
        if state is not None:
            from src.serving.workload.traced_sglang import SGLangEagleChainState
            if state.request_ids != tuple(request.request_id for request in scheduled):
                raise ValueError("SGLang source-state request order differs from admission")
            if isinstance(state, SGLangEagleChainState) and state.phase == "verify":
                if prefill is not None or len(state.draft_slot_mapping) != len(scheduled) * (state.proposal_depth + 1):
                    raise ValueError("SGLang EAGLE verify requires complete per-request source rows")
                live = {request.spec.request_id: request for request in self._running}
                for index, request in enumerate(scheduled):
                    if request.request_id not in self._last_eagle_inputs_by_request:
                        raise ValueError("SGLang verify draft IDs require the preceding declared proposal")
                    previous, previous_row = self._last_eagle_inputs_by_request[request.request_id]
                    if state.draft_token_ids[index * state.proposal_depth] != previous.draft_top1_ids[previous_row]:
                        raise ValueError("SGLang verify first draft IDs must be the preceding declared proposal")
                    expected_bonus_dtype = "int64" if previous.phase == "prefill" else "int32"
                    if state.input_bonus_dtype != expected_bonus_dtype:
                        raise ValueError("SGLang source bonus dtype must match its preceding producer")
                    if state.accept_lens[index] != request.mtp_accepted_tokens + 1:
                        raise ValueError("SGLang declared chain disagrees with Scheduler acceptance")
                    supplied = live[request.request_id]
                    output_ids = tuple(state.target_argmax_ids[row]
                        for row in state.accept_index[index][:state.accept_lens[index]])
                    emitted = generated[request.request_id]
                    if supplied.output_token_ids[supplied.output_produced:supplied.output_produced + emitted] != output_ids[:emitted]:
                        raise ValueError("SGLang declared tree outputs disagree with the external token transcript")
            elif isinstance(state, SGLangEagleChainState) and state.phase == "prefill":
                if prefill is None or len(state.draft_slot_mapping) != sum(request.input_tokens for request in scheduled):
                    raise ValueError("SGLang EAGLE prefill slots must cover the admitted ragged query lengths")
                expected_next = tuple(
                    request.prompt_token_ids[request.prompt_computed + extend]
                    if request.prompt_computed + extend < request.spec.input_tokens else None
                    for request, extend in prefill)
                if any(token is not None for token in expected_next):
                    if (state.prefill_tail_source != "nonfinal_chunk_next_prompt_token"
                            or state.prefill_next_prompt_token_ids != expected_next):
                        raise ValueError("SGLang non-final prefill tail must match the next external prompt token")
                elif (state.prefill_next_prompt_token_ids is not None
                        or any(request.prompt_computed for request, _ in prefill)
                        and state.prefill_tail_source != "final_suffix_no_prompt_override"):
                    raise ValueError("SGLang final prefill suffix requires its explicit no-override source")
                for index, (request, _) in enumerate(prefill):
                    if (generated[request.spec.request_id]
                            and request.output_token_ids[0] != state.bonus_token_ids[index]):
                        raise ValueError("SGLang EAGLE prefill first token disagrees with the external transcript")
        step = BackendStep(
            ScheduleStep(step_id, self.config.pool_name, tuple(scheduled),
                         tuple(finished)),
            generated_tokens=generated,
            embedding_token_ids=(None if scheduled_token_ids is None else tuple(scheduled_token_ids)),
            embedding_token_source=(None if scheduled_token_ids is None else "sglang_explicit_request_token_ids"),
            model_source_inputs=state,
        )
        self._pending = step
        self._step += 1
        return step

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> float:
        if step is not self._pending:
            raise ValueError("completion does not match the pending SGLang step")
        finished = set(step.finished_request_ids)
        if step.model_source_inputs is not None:
            from src.serving.workload.traced_sglang import SGLangEagleChainState
            if isinstance(step.model_source_inputs, SGLangEagleChainState):
                for row, request_id in enumerate(step.model_source_inputs.request_ids):
                    if request_id in finished:
                        if request_id in self._last_eagle_inputs_by_request:
                            del self._last_eagle_inputs_by_request[request_id]
                    else:
                        self._last_eagle_inputs_by_request[request_id] = (step.model_source_inputs, row)
        if self._pending_prefill is not None:
            by_id = {r.spec.request_id: r for r in self._pending_prefill}
            for scheduled in step.schedule.requests:
                request = by_id[scheduled.request_id]
                request.prompt_computed += scheduled.input_tokens
                request.output_produced += step.generated_tokens[
                    scheduled.request_id]
                if (request.prompt_computed == request.spec.input_tokens
                        and scheduled.request_id not in finished):
                    self._running.append(request)
        else:
            for request in self._running:
                request.output_produced += step.generated_tokens[
                    request.spec.request_id]
            before = len(self._running)
            self._running = [
                r for r in self._running if r.spec.request_id not in finished]
            if len(self._running) < before:
                # scheduler.py update_running_batch / filter_batch.
                self._batch_is_full = False
        self._pending = None
        self._pending_prefill = None
        return 0.0
