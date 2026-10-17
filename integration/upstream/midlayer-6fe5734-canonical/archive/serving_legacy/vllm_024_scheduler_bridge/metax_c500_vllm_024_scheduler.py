"""JSONL bridge for C500's audited vLLM 0.24 Scheduler-only control plane.

This module is deliberately executed *inside* the C500 serving container.  It
constructs only ``AsyncScheduler`` plus its metadata-only KV configuration; it
does not create an Engine, a worker, model weights, device tensors, or a CUDA
context.  The caller supplies request arrivals and asks for schedule decisions;
it feeds the synthetic one-token model outputs back only after Midlayer has
evaluated the returned workload.  Consequently this bridge exports Scheduler
facts, never a measured duration.

The wire protocol is intentionally narrow: async, no-prefix-cache, non-MTP,
colocated generation for the two C500 diagnosis cases.  Every record is
fail-closed and has an exact field set so a stale container or client cannot
silently become an approximate vLLM bridge.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_INIT_FIELDS = frozenset({
    "command",
    "model",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "enable_expert_parallel",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "block_size",
    "num_kv_blocks",
    "dummy_token_id",
})
_ADD_FIELDS = frozenset({
    "command",
    "request_id",
    "input_tokens",
    "output_tokens",
    "arrival_time_s",
})
_STEP_FIELDS = frozenset({"command"})
_COMPLETE_FIELDS = frozenset({"command", "step_id"})
_REMOTE_VLLM_VERSION = "0.24.0"


def _record(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, object], fields: frozenset[str], label: str
) -> None:
    if set(value) != fields:
        raise ValueError(
            f"{label} field set differs: missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be non-empty text")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return result


def _source_sha256(subject: object) -> tuple[str, str]:
    path_value = inspect.getsourcefile(subject)
    if path_value is None:
        raise RuntimeError("could not locate vLLM Python source")
    path = Path(path_value)
    if not path.is_file():
        raise RuntimeError(f"vLLM source does not exist: {path}")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_identity() -> dict[str, object]:
    import vllm
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import Request

    if vllm.__version__ != _REMOTE_VLLM_VERSION:
        raise RuntimeError(
            "C500 Scheduler bridge requires vLLM "
            f"{_REMOTE_VLLM_VERSION}, got {vllm.__version__!r}")
    sources = {
        "async_scheduler": _source_sha256(AsyncScheduler),
        "scheduler": _source_sha256(Scheduler),
        "engine_core": _source_sha256(EngineCore),
        "request": _source_sha256(Request),
    }
    return {
        "bridge_contract": "metax_c500_vllm_0_24_scheduler_jsonl_v1",
        "vllm_version": vllm.__version__,
        "sources": {
            name: {"path": path, "sha256": sha256}
            for name, (path, sha256) in sources.items()
        },
    }


@dataclass(frozen=True)
class _RequestShape:
    request_id: str
    input_tokens: int
    output_tokens: int
    arrival_time_s: float


@dataclass(frozen=True)
class _PendingStep:
    step_id: str
    native_output: Any
    raw_generated_tokens: dict[str, int]
    expected_finished_ids: tuple[str, ...]


class _SchedulerOnlyBridge:
    """Exact C500-vLLM-0.24 state machine for the supported diagnosis scope."""

    def __init__(self, init: Mapping[str, object]) -> None:
        _exact_fields(init, _INIT_FIELDS, "init command")
        if init["command"] != "init":
            raise ValueError("first bridge command must be init")
        model = _text(init["model"], "model")
        tensor_parallel_size = _integer(
            init["tensor_parallel_size"], "tensor_parallel_size", minimum=1)
        pipeline_parallel_size = _integer(
            init["pipeline_parallel_size"], "pipeline_parallel_size", minimum=1)
        enable_expert_parallel = init["enable_expert_parallel"]
        if not isinstance(enable_expert_parallel, bool):
            raise TypeError("enable_expert_parallel must be boolean")
        max_model_len = _integer(init["max_model_len"], "max_model_len", minimum=1)
        max_num_batched_tokens = _integer(
            init["max_num_batched_tokens"], "max_num_batched_tokens", minimum=1)
        max_num_seqs = _integer(init["max_num_seqs"], "max_num_seqs", minimum=1)
        block_size = _integer(init["block_size"], "block_size", minimum=1)
        num_kv_blocks = _integer(
            init["num_kv_blocks"], "num_kv_blocks", minimum=1)
        dummy_token_id = _integer(init["dummy_token_id"], "dummy_token_id")
        if pipeline_parallel_size != 1:
            raise NotImplementedError(
                "C500 vLLM 0.24 Scheduler bridge currently requires PP=1")
        if max_num_batched_tokens < max_model_len:
            # Chunked prefill supports this generally, but these diagnosis
            # cases run at 8192/8192 and the bridge must not widen scope.
            raise NotImplementedError(
                "C500 vLLM 0.24 Scheduler bridge requires batch token "
                "budget >= max model length")

        self._identity = _runtime_identity()
        self._dummy_token_id = dummy_token_id
        self._max_model_len = max_model_len
        self._requests: dict[str, _RequestShape] = {}
        self._pending: deque[_PendingStep] = deque()
        self._step_index = 0

        configured_model_path = Path(model).resolve()
        self._temporary_model_directory = None
        if configured_model_path.is_file():
            if configured_model_path.suffix != ".json":
                raise ValueError("Scheduler model config file must use .json")
            self._temporary_model_directory = tempfile.TemporaryDirectory(
                prefix="midlayer-c500-vllm024-model-config-")
            model_directory = Path(self._temporary_model_directory.name)
            (model_directory / "config.json").write_bytes(
                configured_model_path.read_bytes())
        else:
            model_directory = configured_model_path
        if (not model_directory.is_dir()
                or not (model_directory / "config.json").is_file()):
            raise ValueError(
                "Scheduler model must be a JSON config file or a local "
                "directory containing config.json")

        import vllm.platforms as platforms
        from vllm.platforms.cpu import CpuPlatform

        # C500's installed vLLM package selects its MetaX platform at import
        # time.  EngineArgs validation is nevertheless pure control-plane
        # work.  Temporarily request its CPU config branch and restore the
        # platform before the Scheduler is constructed.  This does not load a
        # worker, invoke a device API, or allocate a tensor.
        previous_platform = platforms._current_platform
        platforms._current_platform = CpuPlatform()
        try:
            # ``arg_utils`` captures ``current_platform`` at import time, so
            # set the CPU platform before importing it rather than merely
            # before calling ``create_engine_config``.
            from vllm.engine.arg_utils import EngineArgs

            engine_args = EngineArgs(
                model=str(model_directory),
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
                dtype="bfloat16",
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                block_size=block_size,
                num_gpu_blocks_override=num_kv_blocks,
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                async_scheduling=True,
                enforce_eager=True,
                trust_remote_code=False,
                skip_tokenizer_init=True,
                distributed_executor_backend="uni",
            )
            vllm_config = engine_args.create_engine_config()
        finally:
            platforms._current_platform = previous_platform

        scheduler_config = vllm_config.scheduler_config
        # vLLM 0.24 moved these fields out of SchedulerConfig's public
        # dataclass surface while AsyncScheduler still reads them through its
        # base control-plane contract.  Install exact values from the validated
        # EngineArgs, rather than inheriting a default from a different wheel.
        scheduler_config.max_model_len = max_model_len
        scheduler_config.num_lookahead_slots = 0
        scheduler_config.async_scheduling = True
        vllm_config.cache_config.num_gpu_blocks = num_kv_blocks
        vllm_config.cache_config.num_gpu_blocks_override = num_kv_blocks
        if scheduler_config.max_num_batched_tokens != max_num_batched_tokens:
            raise RuntimeError("vLLM rewrote max_num_batched_tokens")
        if scheduler_config.max_num_seqs != max_num_seqs:
            raise RuntimeError("vLLM rewrote max_num_seqs")
        if scheduler_config.enable_chunked_prefill is not True:
            raise RuntimeError("vLLM disabled requested chunked prefill")
        if vllm_config.cache_config.enable_prefix_caching is not False:
            raise RuntimeError("vLLM enabled prefix caching in a no-cache bridge")
        if vllm_config.cache_config.block_size != block_size:
            raise RuntimeError("vLLM rewrote KV block size")

        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
        )
        from vllm.v1.structured_output import StructuredOutputManager

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        num_layers = model_config.get_num_layers(parallel_config)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        if num_layers < 1 or num_kv_heads < 1 or head_size < 1:
            raise RuntimeError("vLLM reported invalid Qwen KV geometry")
        cache_spec = FullAttentionSpec(
            block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=model_config.dtype,
        )
        layer_names = [
            f"model.layers.{layer_index}.self_attn.attn"
            for layer_index in range(num_layers)
        ]
        page_size_bytes = cache_spec.page_size_bytes
        kv_cache_config = KVCacheConfig(
            num_blocks=num_kv_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=num_kv_blocks * page_size_bytes,
                    shared_by=[layer_name],
                )
                for layer_name in layer_names
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=cache_spec)
            ],
        )
        self._scheduler = AsyncScheduler(
            vllm_config,
            kv_cache_config,
            StructuredOutputManager(vllm_config),
            block_size,
        )
        self._ready = {
            "kind": "ready",
            "runtime": self._identity,
            "effective": {
                "max_model_len": max_model_len,
                "max_num_batched_tokens": max_num_batched_tokens,
                "max_num_seqs": max_num_seqs,
                "block_size": block_size,
                "num_kv_blocks": num_kv_blocks,
                "num_kv_heads": num_kv_heads,
                "head_size": head_size,
                "num_layers": num_layers,
                "attention_kv_layout": "paged_varlen",
            },
        }

    @property
    def ready(self) -> dict[str, object]:
        return self._ready

    def add(self, command: Mapping[str, object]) -> dict[str, object]:
        _exact_fields(command, _ADD_FIELDS, "add command")
        if command["command"] != "add":
            raise ValueError("expected add command")
        request_id = _text(command["request_id"], "request_id")
        if request_id in self._requests:
            raise ValueError(f"duplicate request ID {request_id!r}")
        input_tokens = _integer(command["input_tokens"], "input_tokens", minimum=1)
        output_tokens = _integer(command["output_tokens"], "output_tokens")
        arrival_time_s = _finite_number(command["arrival_time_s"], "arrival_time_s")
        if input_tokens + output_tokens > self._max_model_len:
            raise ValueError("request exceeds max_model_len")

        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        native = Request(
            request_id=request_id,
            prompt_token_ids=[self._dummy_token_id] * input_tokens,
            sampling_params=SamplingParams(max_tokens=max(1, output_tokens)),
            pooling_params=None,
            arrival_time=arrival_time_s,
            block_hasher=None,
        )
        if native.block_hashes != []:
            raise RuntimeError("length-only request unexpectedly hashes content")
        self._scheduler.add_request(native)
        self._requests[request_id] = _RequestShape(
            request_id, input_tokens, output_tokens, arrival_time_s)
        return {"kind": "ok"}

    def _block_ids(self, request_id: str) -> tuple[int, ...]:
        blocks = self._scheduler.kv_cache_manager.get_blocks(request_id)
        groups = blocks.get_block_ids()
        if len(groups) != 1:
            raise NotImplementedError(
                "C500 Scheduler bridge requires exactly one KV cache group")
        return tuple(groups[0])

    def plan(self, command: Mapping[str, object]) -> dict[str, object]:
        _exact_fields(command, _STEP_FIELDS, "plan command")
        if command["command"] != "plan":
            raise ValueError("expected plan command")
        decision_ids = tuple(self._scheduler.requests)
        before = {}
        for request_id in decision_ids:
            native = self._scheduler.requests[request_id]
            before[request_id] = {
                "computed_tokens": native.num_computed_tokens,
                "status": native.status.name,
                "block_ids": self._block_ids(request_id),
                "num_tokens": native.num_tokens,
                "output_token_count": len(native.output_token_ids),
                "output_placeholders": native.num_output_placeholders,
            }
        native_output = self._scheduler.schedule()
        scheduled_ids = tuple(native_output.num_scheduled_tokens)
        if not scheduled_ids:
            # vLLM publishes finished IDs in the native SchedulerOutput after
            # the final model result has been consumed.  This is a pure
            # control-plane decision: it has no scheduled request and must not
            # become a zero-duration workload in Midlayer's device timeline.
            finished_ids = tuple(sorted(native_output.finished_req_ids))
            if finished_ids:
                # update_from_output removed normal finished requests from
                # ``self._scheduler.requests`` already.  Keep the external
                # request shape through this notification so the client can
                # validate its logical completion before acknowledging it.
                unknown = set(finished_ids) - set(self._requests)
                if unknown:
                    raise RuntimeError(
                        "vLLM published a request not admitted by this bridge: "
                        f"{sorted(unknown)!r}")
                for request_id in finished_ids:
                    del self._requests[request_id]
                return {
                    "kind": "finished",
                    "finished_request_ids": list(finished_ids),
                }
            if self._pending:
                return {
                    "kind": "wait",
                    "pending_step_ids": [
                        pending.step_id for pending in self._pending],
                }
            if self._scheduler.has_requests():
                raise RuntimeError("vLLM returned no work with no pending step")
            return {"kind": "idle"}
        if any(request_id not in before for request_id in scheduled_ids):
            raise RuntimeError("vLLM scheduled a request outside decision scope")

        cached = native_output.scheduled_cached_reqs
        resumed_set = set(cached.resumed_req_ids)
        if not resumed_set.issubset(scheduled_ids):
            raise RuntimeError("vLLM resumed an unscheduled request")
        raw_generated_tokens: dict[str, int] = {}
        finished_ids: list[str] = []
        requests: list[dict[str, object]] = []
        for request_id in scheduled_ids:
            count = native_output.num_scheduled_tokens[request_id]
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise RuntimeError("vLLM scheduled invalid token count")
            request = self._requests[request_id]
            native = self._scheduler.requests[request_id]
            before_state = before[request_id]
            computed_before = before_state["computed_tokens"]
            if not isinstance(computed_before, int):
                raise AssertionError("native computed token state changed type")
            computed_after = native.num_computed_tokens
            if computed_after - count != computed_before:
                raise RuntimeError("vLLM scheduled count disagrees with token state")
            phase = "prefill" if computed_before < request.input_tokens else "decode"
            input_count = (
                min(count, request.input_tokens - computed_before)
                if phase == "prefill" else 0
            )
            output_count = count - input_count
            raw_generated = 0
            visible_generated = 0
            terminal = False
            if computed_after >= before_state["num_tokens"]:
                generated_before = (
                    before_state["output_token_count"]
                    + before_state["output_placeholders"])
                remaining = request.output_tokens - generated_before
                model_remaining = self._max_model_len - before_state["num_tokens"]
                if remaining <= 0:
                    if before_state["output_placeholders"] <= 0:
                        raise RuntimeError("vLLM scheduled generation without capacity")
                    raw_generated = 1
                else:
                    raw_generated = 1
                    visible_generated = min(1, remaining, model_remaining)
                    terminal = raw_generated >= min(remaining, model_remaining)
                    if terminal:
                        finished_ids.append(request_id)
            raw_generated_tokens[request_id] = raw_generated
            before_blocks = before_state["block_ids"]
            after_blocks = self._block_ids(request_id)
            if not isinstance(before_blocks, tuple):
                raise AssertionError("native blocks changed type")
            if after_blocks[:len(before_blocks)] != before_blocks:
                raise RuntimeError("vLLM replaced retained blocks without preemption")
            requests.append({
                "request_id": request_id,
                "phase": phase,
                "input_tokens": input_count,
                "output_tokens": output_count,
                "computed_tokens": computed_before,
                "resumed_from_preemption": request_id in resumed_set,
                "raw_model_output_tokens": raw_generated,
                "generated_tokens": visible_generated,
                "block_count_before": len(before_blocks),
                "block_count_after": len(after_blocks),
            })
        step_id = f"main:vllm:{self._step_index}"
        self._step_index += 1
        self._pending.append(_PendingStep(
            step_id,
            native_output,
            raw_generated_tokens,
            tuple(finished_ids),
        ))
        return {
            "kind": "step",
            "step_id": step_id,
            "requests": requests,
            "finished_request_ids": finished_ids,
            "preempted_request_ids": [],
            "attention_kv_layout": "paged_varlen",
        }

    def complete(self, command: Mapping[str, object]) -> dict[str, object]:
        _exact_fields(command, _COMPLETE_FIELDS, "complete command")
        if command["command"] != "complete":
            raise ValueError("expected complete command")
        if not self._pending:
            raise RuntimeError("no pending vLLM step")
        pending = self._pending[0]
        if command["step_id"] != pending.step_id:
            raise ValueError("completion is not for the oldest vLLM step")
        from vllm.v1.outputs import ModelRunnerOutput

        request_ids = list(pending.native_output.num_scheduled_tokens)
        if tuple(request_ids) != tuple(pending.raw_generated_tokens):
            raise RuntimeError("vLLM scheduled request order changed before update")
        present_before_update = set(self._scheduler.requests)
        output = ModelRunnerOutput(
            req_ids=request_ids,
            req_id_to_index={
                request_id: index for index, request_id in enumerate(request_ids)
            },
            sampled_token_ids=[
                [self._dummy_token_id] * pending.raw_generated_tokens[request_id]
                for request_id in request_ids
            ],
        )
        self._scheduler.update_from_output(pending.native_output, output)
        released = tuple(sorted(
            request_id for request_id in request_ids
            if request_id in present_before_update
            and request_id not in self._scheduler.requests
        ))
        expected = tuple(sorted(pending.expected_finished_ids))
        if released != expected:
            raise RuntimeError(
                "vLLM finish state disagrees with planned output: "
                f"expected={expected}, actual={released}")
        self._pending.popleft()
        return {"kind": "ok"}

    def has_work(self, command: Mapping[str, object]) -> dict[str, object]:
        _exact_fields(command, _STEP_FIELDS, "has_work command")
        if command["command"] != "has_work":
            raise ValueError("expected has_work command")
        return {"kind": "has_work", "value": self._scheduler.has_requests()}

    def close(self, command: Mapping[str, object]) -> dict[str, object]:
        _exact_fields(command, _STEP_FIELDS, "close command")
        if command["command"] != "close":
            raise ValueError("expected close command")
        if self._pending:
            raise RuntimeError("cannot close bridge with pending vLLM steps")
        if self._scheduler.has_requests():
            raise RuntimeError("cannot close bridge with unfinished requests")
        return {"kind": "closed"}


@dataclass(frozen=True)
class C500VLLM024RuntimeInfo:
    """Provenance for a local, source-bound C500 vLLM 0.24 Scheduler probe.

    This intentionally is not ``VLLMSchedulerRuntimeInfo``: that dataclass
    describes the repository's pinned 0.11 wheel.  A result carrying this
    object is therefore mechanically distinct from an official 0.11 Forward
    evaluation and must remain a candidate until its source snapshot matches
    the capture runtime.
    """

    bridge_contract: str
    version: str
    source_snapshot_status: str
    source_commit: str | None
    source_hashes: tuple[tuple[str, str], ...]
    model_config_sha256: str
    requested_dtype: str
    requested_tensor_parallel_size: int
    requested_pipeline_parallel_size: int
    requested_expert_parallel: bool
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    kv_block_size_tokens: int
    num_kv_blocks: int
    requested_async_scheduling: bool
    effective_async_scheduling: bool
    requested_chunked_prefill: bool
    effective_chunked_prefill: bool
    requested_prefix_caching: bool
    effective_prefix_caching: bool
    attention_kv_layout: str
    execution_location: str


class C500VLLM024SchedulerControlPlane:
    """Normalized client for the real C500 vLLM 0.24 Scheduler source.

    The bridge process imports only the C500 Python Scheduler source under a
    CPU platform.  This client translates its schedule records into Midlayer's
    normalized :class:`BackendStep`; the caller continues to use the ordinary
    Forward workload construction and kernel predictor.
    """

    _COMMAND_TIMEOUT_S = 60.0

    def __init__(
        self,
        config: Any,
        scheduler_config: Any,
        *,
        source_root: str | Path,
        conda_environment: str = "r2r_qwen3_next",
    ) -> None:
        from src.serving.backend.vllm import VLLMSchedulerConfig
        from src.serving.schedule.backend import BackendConfig

        if not isinstance(config, BackendConfig):
            raise TypeError("C500 vLLM 0.24 bridge requires BackendConfig")
        if not isinstance(scheduler_config, VLLMSchedulerConfig):
            raise TypeError(
                "C500 vLLM 0.24 bridge requires VLLMSchedulerConfig")
        if config.role != "colocated":
            raise NotImplementedError(
                "C500 vLLM 0.24 bridge supports only colocated serving")
        if scheduler_config.max_model_len is None:
            raise ValueError("C500 vLLM 0.24 bridge requires max_model_len")
        if scheduler_config.dtype != "bfloat16":
            raise NotImplementedError(
                "C500 vLLM 0.24 bridge supports only bfloat16")
        if scheduler_config.enable_prefix_caching:
            raise NotImplementedError(
                "C500 vLLM 0.24 bridge supports only no-prefix-cache runs")
        if scheduler_config.mtp_num_speculative_tokens != 0:
            raise NotImplementedError(
                "C500 vLLM 0.24 bridge supports only non-MTP scheduling")
        if not scheduler_config.async_scheduling:
            raise ValueError(
                "C500 vLLM 0.24 bridge requires async scheduling")
        if config.scheduler_pipeline_depth != 2:
            raise NotImplementedError(
                "C500 vLLM 0.24 bridge requires pipeline depth two")
        if (not isinstance(conda_environment, str)
                or not conda_environment):
            raise ValueError("C500 Scheduler conda environment must be text")
        source_directory = Path(source_root).resolve()
        if not source_directory.is_dir():
            raise ValueError("C500 Scheduler source root does not exist")
        if not (source_directory / "vllm" / "__init__.py").is_file():
            raise ValueError("C500 Scheduler source root has no vllm package")
        model_path = Path(scheduler_config.model).resolve()
        if not model_path.is_file() or model_path.suffix != ".json":
            raise ValueError(
                "C500 Scheduler bridge requires a local model JSON config")

        self.config = config
        self.scheduler_config = scheduler_config
        self._source_root = source_directory
        self._closed = False
        self._pending_steps: deque[tuple[Any, str]] = deque()
        self._live_request_ids: set[str] = set()
        self._completed_terminal_ids: set[str] = set()
        self._debug_history: deque[str] = deque(maxlen=64)
        self._terminal_drain_pending = False
        self._repo_root = Path(__file__).resolve().parents[3]
        environment = dict(os.environ)
        environment["VLLM_PLUGINS"] = ""
        existing_pythonpath = (
            environment["PYTHONPATH"] if "PYTHONPATH" in environment else None)
        pythonpath_entries = (str(source_directory), str(self._repo_root))
        if existing_pythonpath:
            environment["PYTHONPATH"] = os.pathsep.join(
                (*pythonpath_entries, existing_pythonpath))
        else:
            environment["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        command = (
            "conda", "run", "--no-capture-output", "-n",
            conda_environment, "python", "-m",
            "src.serving.backend.metax_c500_vllm_024_scheduler",
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=self._repo_root,
            env=environment,
        )
        try:
            ready = self._rpc({
                "command": "init",
                "model": str(model_path),
                "tensor_parallel_size": scheduler_config.tensor_parallel_size,
                "pipeline_parallel_size": scheduler_config.pipeline_parallel_size,
                "enable_expert_parallel": scheduler_config.enable_expert_parallel,
                "max_model_len": scheduler_config.max_model_len,
                "max_num_batched_tokens": config.max_batched_tokens,
                "max_num_seqs": config.max_requests,
                "block_size": config.kv_block_size_tokens,
                "num_kv_blocks": config.num_kv_blocks,
                "dummy_token_id": scheduler_config.dummy_token_id,
            })
            self.runtime_info = self._parse_ready(ready, model_path)
        except BaseException:
            self._force_close()
            raise

    @staticmethod
    def _integer(value: object, label: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"C500 Scheduler {label} must be an integer")
        if value < minimum:
            raise ValueError(f"C500 Scheduler {label} must be >= {minimum}")
        return value

    @staticmethod
    def _text(value: object, label: str) -> str:
        if not isinstance(value, str) or not value:
            raise TypeError(f"C500 Scheduler {label} must be non-empty text")
        return value

    @staticmethod
    def _mapping(value: object, label: str) -> Mapping[str, object]:
        if not isinstance(value, dict):
            raise TypeError(f"C500 Scheduler {label} must be an object")
        return value

    @staticmethod
    def _exact_fields(
        value: Mapping[str, object], fields: frozenset[str], label: str
    ) -> None:
        if set(value) != fields:
            raise ValueError(
                f"C500 Scheduler {label} fields differ: "
                f"missing={sorted(fields - set(value))}, "
                f"extra={sorted(set(value) - fields)}")

    def _rpc(self, record: Mapping[str, object]) -> Mapping[str, object]:
        if self._closed:
            raise RuntimeError("C500 Scheduler bridge is closed")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("C500 Scheduler bridge pipes are unavailable")
        if self._process.poll() is not None:
            raise RuntimeError(
                "C500 Scheduler bridge exited before responding: "
                f"exit={self._process.returncode}")
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        self._process.stdin.write(encoded + "\n")
        self._process.stdin.flush()
        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("C500 Scheduler bridge exited before response")
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "C500 Scheduler bridge returned invalid JSON") from error
        return self._mapping(response, "response")

    def _parse_ready(
        self, ready: Mapping[str, object], model_path: Path
    ) -> C500VLLM024RuntimeInfo:
        self._exact_fields(ready, frozenset({"kind", "runtime", "effective"}),
                           "ready response")
        if ready["kind"] != "ready":
            raise RuntimeError("C500 Scheduler bridge did not become ready")
        runtime = self._mapping(ready["runtime"], "runtime")
        self._exact_fields(
            runtime,
            frozenset({"bridge_contract", "vllm_version", "sources"}),
            "runtime")
        if runtime["bridge_contract"] != (
                "metax_c500_vllm_0_24_scheduler_jsonl_v1"):
            raise RuntimeError("C500 Scheduler bridge contract mismatched")
        if runtime["vllm_version"] != _REMOTE_VLLM_VERSION:
            raise RuntimeError("C500 Scheduler bridge vLLM version mismatched")
        sources = self._mapping(runtime["sources"], "runtime sources")
        expected_sources = frozenset({
            "async_scheduler", "scheduler", "engine_core", "request"})
        self._exact_fields(sources, expected_sources, "runtime sources")
        source_hashes = []
        for source_name in sorted(expected_sources):
            source = self._mapping(sources[source_name], source_name)
            self._exact_fields(source, frozenset({"path", "sha256"}), source_name)
            path = Path(self._text(source["path"], f"{source_name} path"))
            if self._source_root not in path.parents:
                raise RuntimeError("C500 Scheduler source escaped configured root")
            source_hashes.append((source_name, self._text(
                source["sha256"], f"{source_name} SHA256")))
        effective = self._mapping(ready["effective"], "effective config")
        required_effective = frozenset({
            "max_model_len", "max_num_batched_tokens", "max_num_seqs",
            "block_size", "num_kv_blocks", "num_kv_heads", "head_size",
            "num_layers", "attention_kv_layout"})
        self._exact_fields(effective, required_effective, "effective config")
        expected_effective = {
            "max_model_len": self.scheduler_config.max_model_len,
            "max_num_batched_tokens": self.config.max_batched_tokens,
            "max_num_seqs": self.config.max_requests,
            "block_size": self.config.kv_block_size_tokens,
            "num_kv_blocks": self.config.num_kv_blocks,
            "attention_kv_layout": "paged_varlen",
        }
        for field, expected in expected_effective.items():
            if effective[field] != expected:
                raise RuntimeError(
                    f"C500 Scheduler effective {field} disagrees with request")
        for field in ("num_kv_heads", "head_size", "num_layers"):
            self._integer(effective[field], field, minimum=1)
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        return C500VLLM024RuntimeInfo(
            bridge_contract=str(runtime["bridge_contract"]),
            version=str(runtime["vllm_version"]),
            source_snapshot_status="current_runtime_source_not_capture_exact",
            source_commit=None,
            source_hashes=tuple(source_hashes),
            model_config_sha256=model_sha256,
            requested_dtype=self.scheduler_config.dtype,
            requested_tensor_parallel_size=self.scheduler_config.tensor_parallel_size,
            requested_pipeline_parallel_size=(
                self.scheduler_config.pipeline_parallel_size),
            requested_expert_parallel=self.scheduler_config.enable_expert_parallel,
            max_model_len=self.scheduler_config.max_model_len,
            max_num_batched_tokens=self.config.max_batched_tokens,
            max_num_seqs=self.config.max_requests,
            kv_block_size_tokens=self.config.kv_block_size_tokens,
            num_kv_blocks=self.config.num_kv_blocks,
            requested_async_scheduling=True,
            effective_async_scheduling=True,
            requested_chunked_prefill=self.scheduler_config.enable_chunked_prefill,
            effective_chunked_prefill=True,
            requested_prefix_caching=False,
            effective_prefix_caching=False,
            attention_kv_layout=str(effective["attention_kv_layout"]),
            execution_location="local_cpu_scheduler_only",
        )

    def add_request(self, request: Any) -> None:
        from src.serving.request_gen.request_stream import RequestSpec

        if not isinstance(request, RequestSpec):
            raise TypeError("C500 Scheduler request must be RequestSpec")
        if request.request_id in self._live_request_ids:
            raise ValueError(f"duplicate C500 Scheduler request {request.request_id!r}")
        if request.prefix_hit_tokens != 0:
            raise NotImplementedError(
                "C500 vLLM 0.24 bridge refuses prefix-cache requests")
        response = self._rpc({
            "command": "add",
            "request_id": request.request_id,
            "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens,
            "arrival_time_s": request.arrival_time_s,
        })
        self._exact_fields(response, frozenset({"kind"}), "add response")
        if response["kind"] != "ok":
            raise RuntimeError("C500 Scheduler declined request admission")
        self._live_request_ids.add(request.request_id)

    @property
    def debug_history(self) -> tuple[str, ...]:
        """Recent protocol events for the diagnostic runner's failure report."""

        return tuple(self._debug_history)

    def add_requests(self, requests: tuple[Any, ...]) -> None:
        if not isinstance(requests, tuple):
            raise TypeError("C500 Scheduler bulk requests must be a tuple")
        for request in requests:
            self.add_request(request)

    def _cache_event(
        self,
        request_id: str,
        action: Any,
        start: int,
        end: int,
    ) -> Any:
        from src.serving.kv_cache.kv_cache_events import KVCacheEvent

        return KVCacheEvent(
            request_id=request_id,
            action=action,
            tier="hbm",
            token_start=start,
            token_end=end,
            bytes=(end - start) * self.config.kv_bytes_per_token,
        )

    def _step_from_record(self, record: Mapping[str, object]) -> Any:
        from src.serving.kv_cache.kv_cache_events import KVCacheAction
        from src.serving.schedule.schedule_step import (
            BackendStep, ScheduledRequest, ScheduleStep,
        )

        self._exact_fields(record, frozenset({
            "kind", "step_id", "requests", "finished_request_ids",
            "preempted_request_ids", "attention_kv_layout"}), "step response")
        if record["kind"] != "step":
            raise RuntimeError("C500 Scheduler step record has wrong kind")
        step_id = self._text(record["step_id"], "step ID")
        requests_value = record["requests"]
        if not isinstance(requests_value, list) or not requests_value:
            raise ValueError("C500 Scheduler step needs scheduled requests")
        finished_value = record["finished_request_ids"]
        preempted_value = record["preempted_request_ids"]
        if not isinstance(finished_value, list) or not isinstance(preempted_value, list):
            raise TypeError("C500 Scheduler completion sets must be lists")
        finished_ids = tuple(self._text(value, "finished request ID")
                             for value in finished_value)
        preempted_ids = tuple(self._text(value, "preempted request ID")
                              for value in preempted_value)
        if record["attention_kv_layout"] != "paged_varlen":
            raise RuntimeError("C500 Scheduler returned unsupported KV layout")
        scheduled_requests = []
        generated_tokens: dict[str, int] = {}
        cache_events = []
        fields = frozenset({
            "request_id", "phase", "input_tokens", "output_tokens",
            "computed_tokens", "resumed_from_preemption",
            "raw_model_output_tokens", "generated_tokens",
            "block_count_before", "block_count_after"})
        for raw_request in requests_value:
            item = self._mapping(raw_request, "scheduled request")
            self._exact_fields(item, fields, "scheduled request")
            request_id = self._text(item["request_id"], "request ID")
            if request_id not in self._live_request_ids:
                raise RuntimeError("C500 Scheduler scheduled unknown request")
            phase = item["phase"]
            if phase not in ("prefill", "decode"):
                raise ValueError("C500 Scheduler returned unknown phase")
            input_tokens = self._integer(item["input_tokens"], "input tokens")
            output_tokens = self._integer(item["output_tokens"], "output tokens")
            computed_tokens = self._integer(
                item["computed_tokens"], "computed tokens")
            raw_model_output_tokens = self._integer(
                item["raw_model_output_tokens"], "raw output tokens")
            visible_generated = self._integer(
                item["generated_tokens"], "visible generated tokens")
            before_blocks = self._integer(
                item["block_count_before"], "before block count")
            after_blocks = self._integer(
                item["block_count_after"], "after block count")
            resumed = item["resumed_from_preemption"]
            if not isinstance(resumed, bool):
                raise TypeError("C500 Scheduler resumed flag must be boolean")
            if after_blocks < before_blocks:
                raise RuntimeError("C500 Scheduler released blocks during schedule")
            if raw_model_output_tokens not in (0, 1):
                raise RuntimeError("non-MTP C500 Scheduler emitted invalid raw output")
            if visible_generated > raw_model_output_tokens:
                raise RuntimeError("C500 Scheduler visible output exceeds raw output")
            scheduled_requests.append(ScheduledRequest(
                request_id=request_id,
                phase=phase,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                computed_tokens=computed_tokens,
                resumed_from_preemption=resumed,
                raw_model_output_tokens=raw_model_output_tokens,
            ))
            generated_tokens[request_id] = visible_generated
            if after_blocks > before_blocks:
                cache_events.append(self._cache_event(
                    request_id, KVCacheAction.ALLOCATE,
                    before_blocks * self.config.kv_block_size_tokens,
                    after_blocks * self.config.kv_block_size_tokens))
            if input_tokens:
                cache_events.append(self._cache_event(
                    request_id, KVCacheAction.MISS,
                    computed_tokens, computed_tokens + input_tokens))
            if request_id in finished_ids and after_blocks:
                cache_events.append(self._cache_event(
                    request_id, KVCacheAction.RELEASE,
                    0, after_blocks * self.config.kv_block_size_tokens))
        step = BackendStep(
            schedule=ScheduleStep(
                step_id=step_id,
                pool_name=self.config.pool_name,
                requests=tuple(scheduled_requests),
                finished_request_ids=finished_ids,
                preempted_request_ids=preempted_ids,
                attention_kv_layout="paged_varlen",
            ),
            kv_cache_events=tuple(cache_events),
            generated_tokens=generated_tokens,
        )
        return step

    def plan_step(self, virtual_time_s: float) -> Any:
        if isinstance(virtual_time_s, bool) or not isinstance(
                virtual_time_s, (int, float)):
            raise TypeError("C500 Scheduler virtual time must be numeric")
        while True:
            response = self._rpc({"command": "plan"})
            kind = response["kind"] if "kind" in response else None
            if kind == "step":
                step = self._step_from_record(response)
                self._pending_steps.append((step, step.schedule.step_id))
                self._debug_history.append(
                    f"plan:{step.schedule.step_id}:"
                    f"finished={step.schedule.finished_request_ids!r}:"
                    f"local_pending={tuple(item[1] for item in self._pending_steps)!r}")
                return step
            if kind == "wait":
                self._exact_fields(
                    response, frozenset({"kind", "pending_step_ids"}),
                    "wait response")
                pending_ids = response["pending_step_ids"]
                if not isinstance(pending_ids, list) or not pending_ids:
                    raise RuntimeError("C500 Scheduler wait has no pending step")
                normalized_pending_ids = tuple(
                    self._text(value, "wait pending step ID")
                    for value in pending_ids)
                local_pending_ids = tuple(
                    local_step_id for _, local_step_id in self._pending_steps)
                if normalized_pending_ids != local_pending_ids:
                    raise RuntimeError(
                        "C500 Scheduler pending steps disagree with client: "
                        f"remote={normalized_pending_ids!r}, "
                        f"local={local_pending_ids!r}")
                self._debug_history.append(
                    f"wait:remote={normalized_pending_ids!r}:"
                    f"local={local_pending_ids!r}")
                return None
            if kind == "finished":
                self._exact_fields(
                    response, frozenset({"kind", "finished_request_ids"}),
                    "finished response")
                finished = response["finished_request_ids"]
                if not isinstance(finished, list) or not finished:
                    raise RuntimeError("C500 Scheduler returned empty finish notice")
                for request_id in finished:
                    normalized_request_id = self._text(
                        request_id, "finished request ID")
                    if normalized_request_id not in self._completed_terminal_ids:
                        raise RuntimeError(
                            "C500 Scheduler published an unplanned finish")
                self._debug_history.append(
                    f"finished:{tuple(finished)!r}:"
                    f"local_pending={tuple(item[1] for item in self._pending_steps)!r}")
                continue
            if kind == "idle":
                self._exact_fields(response, frozenset({"kind"}), "idle response")
                self._debug_history.append("idle")
                return None
            raise RuntimeError("C500 Scheduler returned unknown plan response")

    def complete_step(
        self,
        step: Any,
        virtual_time_s: float,
        *,
        execute_model: bool,
        execute_transfers: bool,
        sample_tokens: bool,
    ) -> float:
        if execute_model or execute_transfers or sample_tokens:
            raise ValueError("C500 Scheduler bridge cannot execute a data plane")
        if not self._pending_steps or self._pending_steps[0][0] is not step:
            raise ValueError("completion does not match oldest C500 Scheduler step")
        _, step_id = self._pending_steps[0]
        response = self._rpc({"command": "complete", "step_id": step_id})
        self._exact_fields(response, frozenset({"kind"}), "complete response")
        if response["kind"] != "ok":
            raise RuntimeError("C500 Scheduler rejected completion")
        self._pending_steps.popleft()
        finished_ids = set(step.schedule.finished_request_ids)
        if not finished_ids.issubset(self._live_request_ids):
            raise RuntimeError("C500 Scheduler completed an unknown request")
        self._live_request_ids.difference_update(finished_ids)
        self._completed_terminal_ids.update(finished_ids)
        if finished_ids and not self._live_request_ids:
            self._terminal_drain_pending = True
        self._debug_history.append(
            f"complete:{step_id}:"
            f"local_pending={tuple(item[1] for item in self._pending_steps)!r}")
        return 0.0

    def has_work(self) -> bool:
        response = self._rpc({"command": "has_work"})
        self._exact_fields(response, frozenset({"kind", "value"}),
                           "has-work response")
        if response["kind"] != "has_work" or not isinstance(response["value"], bool):
            raise RuntimeError("C500 Scheduler returned invalid work state")
        has_work = response["value"]
        if (not self._terminal_drain_pending or self._pending_steps
                or not has_work):
            return has_work
        # EngineCore leaves the final completion notification to its next
        # scheduler turn.  Midlayer's event loop asks ``has_work`` before that
        # turn and, correctly, expects a work-bearing BackendStep if it sees
        # true.  Consume only this all-requests-terminal bookkeeping turn so
        # it cannot be mistaken for an unmodelled device step.
        terminal_plan = self.plan_step(0.0)
        if terminal_plan is not None:
            raise RuntimeError(
                "C500 Scheduler emitted device work after all requests "
                "were marked terminal")
        if self._live_request_ids:
            raise RuntimeError(
                "C500 Scheduler terminal drain retained live requests")
        settled = self._rpc({"command": "has_work"})
        self._exact_fields(settled, frozenset({"kind", "value"}),
                           "settled has-work response")
        if settled["kind"] != "has_work" or settled["value"] is not False:
            raise RuntimeError("C500 Scheduler terminal drain did not settle")
        self._terminal_drain_pending = False
        self._debug_history.append("terminal-drain-settled")
        return False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._pending_steps:
                raise RuntimeError("cannot close C500 Scheduler with pending steps")
            if self._live_request_ids:
                raise RuntimeError("cannot close C500 Scheduler with live requests")
            response = self._rpc({"command": "close"})
            self._exact_fields(response, frozenset({"kind"}), "close response")
            if response["kind"] != "closed":
                raise RuntimeError("C500 Scheduler close failed")
        finally:
            self._force_close()

    def _force_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5.0)


def _read_record() -> Mapping[str, object] | None:
    line = sys.stdin.readline()
    if line == "":
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSONL request: {error}") from error
    return _record(parsed, "JSONL request")


def _write_record(value: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    init = _read_record()
    if init is None:
        raise RuntimeError("C500 Scheduler bridge requires an init record")
    # vLLM's Python and plugin bootstrap logs otherwise share stdout with the
    # JSONL protocol.  Bind their handlers to stderr only while vLLM is first
    # imported/configured; protocol records always use the original stdout.
    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        bridge = _SchedulerOnlyBridge(init)
    finally:
        sys.stdout = protocol_stdout
    _write_record(bridge.ready)
    while True:
        command = _read_record()
        if command is None:
            raise RuntimeError("C500 Scheduler bridge stdin closed before close")
        command_name = command["command"] if "command" in command else None
        if command_name == "add":
            response = bridge.add(command)
        elif command_name == "plan":
            response = bridge.plan(command)
        elif command_name == "complete":
            response = bridge.complete(command)
        elif command_name == "has_work":
            response = bridge.has_work(command)
        elif command_name == "close":
            response = bridge.close(command)
            _write_record(response)
            return
        else:
            raise ValueError(f"unknown C500 Scheduler bridge command: {command_name!r}")
        _write_record(response)


if __name__ == "__main__":
    main()
