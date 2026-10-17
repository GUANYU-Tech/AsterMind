"""SGLang scheduler/RadixCache bridge with a short-circuited model runner.

SGLang does not expose a worker-free ``Scheduler`` constructor: constructing
one also constructs ``TpModelWorker`` and loads the model.  Consequently this
module deliberately does not offer a misleading ``from_model`` API.  The
production bridge is :meth:`SGLangBackend.from_scheduler`, which takes an
already initialized, idle SGLang scheduler, reuses its scheduling and cache
state, and replaces only model execution/sampling with a CPU dummy result.

All SGLang imports are lazy.  Midlayer therefore remains importable when
SGLang is not installed, and embedding applications can still supply an
already-normalized control plane through the regular constructor.
"""

from __future__ import annotations

from typing import Any

from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.request_gen.request_stream import RequestSpec
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.schedule_step import BackendStep, ScheduleStep, ScheduledRequest


def _sglang_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"SGLang {field} must be an integer")
    return value


def _sglang_request_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("SGLang request ID must be a non-empty string")
    return value


class SGLangBackend:
    """Framework-independent facade over an SGLang control plane."""

    def __init__(self, config: BackendConfig, control_plane: Any) -> None:
        self.config = config
        self._control_plane = control_plane

    @classmethod
    def from_scheduler(
        cls,
        config: BackendConfig,
        scheduler: Any,
        *,
        dummy_token_id: int = 42,
    ) -> "SGLangBackend":
        """Bridge an initialized SGLang ``Scheduler`` without running a model.

        The scheduler must be idle and must not have its event loop running.
        SGLang overlap/speculative and native disaggregation modes are rejected
        because their output pipelines require worker-owned GPU state or a
        live KV-transfer process.  Colocated, non-overlap generation uses the
        real ``ScheduleBatch`` and RadixCache decisions.
        """

        if isinstance(dummy_token_id, bool) or not isinstance(
                dummy_token_id, int):
            raise TypeError("SGLang dummy_token_id must be an integer")
        return cls(
            config,
            _NativeSGLangControlPlane(
                config,
                scheduler,
                dummy_token_id=dummy_token_id,
            ),
        )

    def add_request(self, request: RequestSpec) -> None:
        if request.output_tokens == 0:
            raise NotImplementedError(
                "SGLang zero-output request completion is not implemented")
        if request.prefix_hit_tokens != 0:
            raise NotImplementedError(
                "SGLang trace-authoritative prefix injection is not implemented")
        self._control_plane.add_request(request)

    def plan_step(self, virtual_time_s: float):
        return self._control_plane.plan_step(virtual_time_s)

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> float:
        self._control_plane.complete_step(
            step,
            virtual_time_s,
            execute_model=False,
            execute_transfers=False,
            sample_tokens=False,
        )

    def has_work(self) -> bool:
        result = self._control_plane.has_work()
        if not isinstance(result, bool):
            raise TypeError("SGLang control_plane.has_work() must return boolean")
        return result

    def close(self) -> None:
        if not hasattr(self._control_plane, "close"):
            return
        self._control_plane.close()


class _NativeSGLangControlPlane:
    def __init__(
        self,
        config: BackendConfig,
        scheduler: Any,
        *,
        dummy_token_id: int,
    ) -> float:
        _validate_native_scheduler(scheduler)
        self.config = config
        self.scheduler = scheduler
        self.dummy_token_id = dummy_token_id
        self._requests: dict[str, RequestSpec] = {}
        self._native_requests: dict[str, Any] = {}
        self._first_schedule: set[str] = set()
        self._pending: tuple[BackendStep, Any] | None = None
        self._step_index = 0

    def add_request(self, request: RequestSpec) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"duplicate SGLang request ID {request.request_id!r}")
        if request.output_tokens == 0:
            raise NotImplementedError(
                "SGLang zero-output request completion is not implemented")
        if request.prefix_hit_tokens != 0:
            raise NotImplementedError(
                "SGLang trace-authoritative prefix injection is not implemented")
        output_limit = 1 if self.config.is_prefill_only else request.output_tokens
        native = _make_native_request(
            self.scheduler,
            request,
            dummy_token_id=self.dummy_token_id,
            max_output_tokens=output_limit,
        )
        self.scheduler._add_request_to_queue(native)
        if native.finished():
            raise RuntimeError(
                f"SGLang rejected request {request.request_id!r} during admission"
            )
        self._requests[request.request_id] = request
        self._native_requests[request.request_id] = native

    def plan_step(self, virtual_time_s: float) -> BackendStep | None:
        if self._pending is not None:
            raise RuntimeError("SGLang step must be completed before planning another step")
        if not self.has_work():
            return None

        batch = self.scheduler.get_next_batch_to_run()
        if batch is None:
            raise RuntimeError("SGLang Scheduler returned no batch while work remains")
        self.scheduler.cur_batch = batch

        requests, generated_tokens, finished_request_ids = (
            _normalize_native_batch(batch)
        )
        cache_events = []
        for scheduled in requests:
            request_id = scheduled.request_id
            if request_id in self._first_schedule:
                continue
            self._first_schedule.add(request_id)
            native = self._native_requests[request_id]
            request = self._requests[request_id]
            cache_events.extend(self._initial_cache_events(native, request))

        step = BackendStep(
            schedule=ScheduleStep(
                step_id=f"{self.config.pool_name}:sglang:{self._step_index}",
                pool_name=self.config.pool_name,
                requests=tuple(requests),
                finished_request_ids=tuple(finished_request_ids),
            ),
            kv_cache_events=tuple(cache_events),
            generated_tokens=generated_tokens,
        )
        self._pending = (step, batch)
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
    ) -> None:
        del virtual_time_s
        if execute_model or execute_transfers or sample_tokens:
            raise ValueError(
                "the SGLang simulator bridge never executes a data plane")
        if self._pending is None or self._pending[0] is not step:
            raise ValueError("completion does not match the pending SGLang step")

        _, batch = self._pending
        result = _fake_generation_batch_result(
            batch,
            self.dummy_token_id,
            step.generated_tokens,
        )
        self.scheduler.process_batch_result(batch, result)
        # These assignments mirror ``Scheduler.event_loop_normal`` around
        # process_batch_result.  The next call uses last_batch to merge/filter
        # native running state before selecting another batch.
        self.scheduler.last_batch = batch
        self.scheduler.cur_batch = None
        self._pending = None
        for request_id in tuple(self._native_requests):
            if self._native_requests[request_id].finished():
                del self._native_requests[request_id]
                del self._requests[request_id]
        return 0.0

    def has_work(self) -> bool:
        return self._pending is not None or bool(self._requests)

    def _initial_cache_events(
        self,
        native_request: Any,
        request: RequestSpec,
    ) -> list[KVCacheEvent]:
        prompt_tokens = request.input_tokens
        tiered_fields = (
            "cached_tokens_device", "cached_tokens_host",
            "cached_tokens_storage",
        )
        has_tiered = [hasattr(native_request, name) for name in tiered_fields]
        if all(has_tiered):
            device = _sglang_integer(
                native_request.cached_tokens_device, "device cached tokens")
            host = _sglang_integer(
                native_request.cached_tokens_host, "host cached tokens")
            storage = _sglang_integer(
                native_request.cached_tokens_storage, "storage cached tokens")
            if not hasattr(native_request, "cached_tokens"):
                raise TypeError("SGLang tiered request lacks cached_tokens total")
            cached_total = _sglang_integer(
                native_request.cached_tokens, "total cached tokens")
            if min(device, host, storage, cached_total) < 0:
                raise ValueError("SGLang cached-token counts cannot be negative")
            if device + host + storage != cached_total:
                raise ValueError(
                    "SGLang tiered cached-token counts disagree with cached_tokens")
        elif any(has_tiered):
            raise TypeError(
                "SGLang request exposes an incomplete tiered cache contract")
        elif hasattr(native_request, "cached_tokens"):
            device = _sglang_integer(
                native_request.cached_tokens, "device cached tokens")
            if device < 0:
                raise ValueError("SGLang cached-token count cannot be negative")
            host = 0
            storage = 0
        else:
            raise TypeError("SGLang request lacks cached-token accounting")
        if device + host + storage > prompt_tokens:
            raise ValueError(
                "SGLang cached-token count exceeds the request prompt length")
        if device + host + storage != request.prefix_hit_tokens:
            raise ValueError(
                "SGLang content-derived cached tokens disagree with the "
                "trace-authoritative prefix_hit_tokens")

        events = []
        offset = 0
        for tier, count in (("hbm", device), ("host", host), ("storage", storage)):
            if count <= 0:
                continue
            events.append(self._cache_event(
                request.request_id,
                KVCacheAction.HIT,
                tier,
                offset,
                offset + count,
            ))
            offset += count
        if offset < prompt_tokens:
            events.append(self._cache_event(
                request.request_id,
                KVCacheAction.MISS,
                "hbm",
                offset,
                prompt_tokens,
            ))
        return events

    def _cache_event(
        self,
        request_id: str,
        action: KVCacheAction,
        tier: str,
        start: int,
        end: int,
    ) -> KVCacheEvent:
        return KVCacheEvent(
            request_id=request_id,
            action=action,
            tier=tier,
            token_start=start,
            token_end=end,
            bytes=(end - start) * self.config.kv_bytes_per_token,
        )


def _validate_native_scheduler(scheduler: Any) -> None:
    required_methods = (
        "get_next_batch_to_run",
        "process_batch_result",
        "_add_request_to_queue",
        "init_req_max_new_tokens",
    )
    missing_methods = [
        name for name in required_methods
        if not hasattr(scheduler, name) or not callable(getattr(scheduler, name))
    ]
    if missing_methods:
        raise TypeError(
            "SGLang scheduler is missing native control-plane methods: "
            + ", ".join(missing_methods)
        )
    required_fields = (
        "enable_overlap", "pp_size", "spec_algorithm",
        "disaggregation_mode", "model_config", "dllm_config", "tokenizer",
        "disable_radix_cache",
    )
    missing_fields = [
        name for name in required_fields if not hasattr(scheduler, name)
    ]
    if missing_fields:
        raise TypeError(
            "SGLang scheduler is missing native control-plane fields: "
            + ", ".join(missing_fields)
        )
    if not isinstance(scheduler.enable_overlap, bool):
        raise TypeError("SGLang enable_overlap must be boolean")
    if scheduler.enable_overlap:
        raise ValueError("SGLang overlap scheduling requires worker-owned GPU results")
    if not isinstance(scheduler.disable_radix_cache, bool):
        raise TypeError("SGLang disable_radix_cache must be boolean")
    if not scheduler.disable_radix_cache:
        raise ValueError(
            "SGLang content prefix cache must be disabled for "
            "trace-authoritative prefix semantics")
    pp_size = _sglang_integer(scheduler.pp_size, "pp_size")
    if pp_size != 1:
        raise ValueError("SGLang pipeline-parallel result relaying is not supported")
    spec_algorithm = scheduler.spec_algorithm
    if not hasattr(spec_algorithm, "is_none") or not callable(
            spec_algorithm.is_none):
        raise TypeError("SGLang spec_algorithm lacks is_none()")
    spec_is_none = spec_algorithm.is_none()
    if not isinstance(spec_is_none, bool):
        raise TypeError("SGLang spec_algorithm.is_none() must return boolean")
    if not spec_is_none:
        raise ValueError("SGLang speculative decoding requires a live draft worker")
    disaggregation_mode = scheduler.disaggregation_mode
    if disaggregation_mode is not None:
        if not hasattr(disaggregation_mode, "value"):
            raise TypeError("SGLang disaggregation_mode lacks enum value")
        if not isinstance(disaggregation_mode.value, str):
            raise TypeError("SGLang disaggregation_mode.value must be a string")
        mode = disaggregation_mode.value.lower()
        if mode not in ("", "null", "none"):
            raise ValueError(
                "SGLang native disaggregation requires live KV-transfer processes"
            )


def _make_native_request(
    scheduler: Any,
    request: RequestSpec,
    *,
    dummy_token_id: int,
    max_output_tokens: int,
):
    max_output_tokens = _sglang_integer(
        max_output_tokens, "max_output_tokens")
    if max_output_tokens <= 0:
        raise ValueError("SGLang max_output_tokens must be positive")
    try:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "SGLangBackend.from_scheduler requires an SGLang installation"
        ) from exc

    sampling_params = SamplingParams(
        max_new_tokens=max_output_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    model_config = scheduler.model_config
    if model_config is None or not hasattr(model_config, "vocab_size"):
        raise TypeError("SGLang scheduler model_config lacks vocab_size")
    vocab_size = _sglang_integer(model_config.vocab_size, "model vocab_size")
    if vocab_size <= 0:
        raise ValueError("SGLang model vocab_size must be positive")
    dummy_token_id = _sglang_integer(dummy_token_id, "dummy_token_id")
    if not 0 <= dummy_token_id < vocab_size:
        raise ValueError(
            "SGLang dummy_token_id must be within the model vocabulary")
    native = Req(
        request.request_id,
        "",
        [dummy_token_id] * request.input_tokens,
        sampling_params,
        eos_token_ids=set(),
        vocab_size=vocab_size,
        dllm_config=scheduler.dllm_config,
        disagg_mode=scheduler.disaggregation_mode,
    )
    native.tokenizer = scheduler.tokenizer
    scheduler.init_req_max_new_tokens(native)
    native_max_output_tokens = _sglang_integer(
        native.sampling_params.max_new_tokens,
        "native max_new_tokens",
    )
    if native_max_output_tokens < max_output_tokens:
        raise ValueError(
            "SGLang scheduler capacity truncated max_new_tokens from "
            f"{max_output_tokens} to {native.sampling_params.max_new_tokens}"
        )
    return native


def _normalize_native_batch(
    batch: Any,
) -> tuple[list[ScheduledRequest], dict[str, int], list[str]]:
    """Normalize a real SGLang ``ScheduleBatch`` without importing SGLang."""

    forward_mode = batch.forward_mode
    if not hasattr(forward_mode, "is_decode") or not callable(
            forward_mode.is_decode):
        raise RuntimeError("SGLang forward mode lacks is_decode()")
    if not hasattr(forward_mode, "is_extend") or not callable(
            forward_mode.is_extend):
        raise RuntimeError("SGLang forward mode lacks is_extend()")
    is_decode = forward_mode.is_decode()
    is_extend = forward_mode.is_extend()
    if not isinstance(is_decode, bool) or not isinstance(is_extend, bool):
        raise TypeError("SGLang forward-mode predicates must return booleans")
    if not is_decode and not is_extend:
        raise RuntimeError(
            f"unsupported SGLang forward mode for scheduler simulation: "
            f"{forward_mode}"
        )

    reqs = list(batch.reqs)
    if is_decode:
        prefix_lens = [0] * len(reqs)
        extend_lens = [0] * len(reqs)
        decoding_reqs = {id(req) for req in reqs}
    elif is_extend:
        for field in ("prefix_lens", "extend_lens", "decoding_reqs"):
            if not hasattr(batch, field):
                raise RuntimeError(
                    f"SGLang extend batch lacks required {field}")
        prefix_lens = list(batch.prefix_lens)
        extend_lens = list(batch.extend_lens)
        decoding_reqs = {id(req) for req in batch.decoding_reqs}
    else:
        raise RuntimeError(f"unsupported SGLang forward mode: {forward_mode}")
    if len(prefix_lens) != len(reqs) or len(extend_lens) != len(reqs):
        raise RuntimeError("SGLang ScheduleBatch token vectors do not match its requests")

    scheduled = []
    generated_tokens: dict[str, int] = {}
    finished_request_ids = []
    for index, req in enumerate(reqs):
        request_id = _sglang_request_id(req.rid)
        request_is_decode = is_decode or id(req) in decoding_reqs
        if request_is_decode:
            phase = "decode"
            input_tokens = 0
            output_tokens = 1
            computed_tokens = (
                len(req.origin_input_ids) + len(req.output_ids) - 1)
            if computed_tokens < 0:
                raise ValueError(
                    f"SGLang decode request {request_id!r} has negative "
                    "computed-token count")
            generates_token = True
        else:
            phase = "prefill"
            input_tokens = _sglang_integer(
                extend_lens[index], "extend token count")
            output_tokens = 0
            computed_tokens = _sglang_integer(
                prefix_lens[index], "prefix token count")
            if not hasattr(req, "is_chunked"):
                raise RuntimeError(
                    f"SGLang request {request_id!r} lacks is_chunked")
            is_chunked = _sglang_integer(req.is_chunked, "is_chunked flag")
            if is_chunked not in (0, 1):
                raise ValueError("SGLang is_chunked flag must be 0 or 1")
            generates_token = is_chunked == 0
        if input_tokens + output_tokens <= 0:
            raise RuntimeError(
                f"SGLang scheduled zero tokens for request {request_id!r}"
            )
        scheduled.append(ScheduledRequest(
            request_id=request_id,
            phase=phase,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            computed_tokens=computed_tokens,
            raw_model_output_tokens=int(generates_token),
        ))
        generated_tokens[request_id] = 0

        if generates_token:
            generated_tokens[request_id] = 1
            max_new_tokens = _sglang_integer(
                req.sampling_params.max_new_tokens, "max_new_tokens")
            if max_new_tokens <= 0:
                raise ValueError("SGLang max_new_tokens must be positive")
            if len(req.output_ids) + 1 >= max_new_tokens:
                finished_request_ids.append(request_id)

    return scheduled, generated_tokens, finished_request_ids


def _fake_generation_batch_result(
    batch: Any,
    dummy_token_id: int,
    generated_tokens: dict[str, int],
):
    try:
        import torch
        from sglang.srt.managers.utils import GenerationBatchResult
    except ImportError as exc:
        raise RuntimeError(
            "SGLangBackend.from_scheduler requires an SGLang installation"
        ) from exc

    token_ids = []
    for req in batch.reqs:
        request_id = _sglang_request_id(req.rid)
        generated = _sglang_integer(
            generated_tokens[request_id], "generated token count")
        if generated not in (0, 1):
            raise ValueError(
                "non-speculative SGLang steps must generate zero or one token")
        token_ids.append(dummy_token_id)
    for field in ("extend_lens", "extend_logprob_start_lens"):
        if not hasattr(batch, field):
            raise TypeError(f"SGLang batch lacks {field}")
    return GenerationBatchResult(
        next_token_ids=torch.tensor(token_ids, dtype=torch.long, device="cpu"),
        extend_input_len_per_req=(
            list(batch.extend_lens) if batch.extend_lens is not None
            else None
        ),
        extend_logprob_start_len_per_req=(
            list(batch.extend_logprob_start_lens)
            if batch.extend_logprob_start_lens is not None
            else None
        ),
        can_run_cuda_graph=False,
    )
