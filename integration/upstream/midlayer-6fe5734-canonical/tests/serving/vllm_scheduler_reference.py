"""Independent native-vLLM Scheduler reference for adapter validation.

This harness intentionally does not import or call the Midlayer vLLM adapter.
It constructs the pinned upstream Scheduler directly in a spawned CPU-only
process and records raw scheduler and KV-block state around every step.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType
from typing import Any


_VLLM_VERSION = "0.11.0"
_VLLM_SOURCE_COMMIT = "b8b302cde434df8c9289a2b465406b47ebab1c2d"
_PACKAGE_PYTHON_FILE_COUNT = 945
_PACKAGE_PYTHON_MANIFEST_SHA256 = (
    "49a6f51a5e375a3eabb2dee2042c05efa78d7b03c89235787932c94cc0904bdb"
)
_SOURCE_HASHES = (
    ("v1/core/sched/scheduler.py",
     "357ac414a5b5da8c3423984dd1fdde860584a0d417c30ac2ee25874ea9e5a04c"),
    ("v1/core/kv_cache_utils.py",
     "0824beae09a3a8faeb54e8e0605d1661045e34a33a2026de537e6849a4c78ee4"),
    ("v1/kv_cache_interface.py",
     "bf0d73db3442b5bfe96c558c36cd5601e800aea9a1508c989ffe4b54e8d5ce98"),
    ("v1/request.py",
     "be457706757fe6734c5a1bd9f3261722276ec27616dac996d6b1c33cacdfa5ec"),
    ("v1/outputs.py",
     "b859de3c3e057d5c38ce050b35b2bb5b222a4050dbd83f43c3cd1f3b15018124"),
    ("engine/arg_utils.py",
     "2238ade872b5f7478330fa9ca1214389520e08991582090e3e03e2feabcf95ac"),
    ("platforms/__init__.py",
     "48bf2526618f5f435394dfd4ba52cf21ee71acfd1d0eb4bf45ba3c784727dcc7"),
    ("platforms/cpu.py",
     "0a71e9e78a5feb4435884fac762b2718730c2463c961a88200df4ed586d35cd2"),
    ("config/speculative.py",
     "dc025a9d3002e80737046b1efdbc4a70a5f0bc5d119104b3117c13ec033a87ed"),
    ("v1/spec_decode/eagle.py",
     "a2857fd3a3a6c261892b3b25655b513428399a0d3a763d77fcbd52d7105b6ebf"),
    ("v1/worker/gpu_model_runner.py",
     "ea6b7bfe4082349e80042df4801c7ef9e18c80d1c3879eeb6ac6e9c476303574"),
    ("v1/engine/core.py",
     "4573563d926cafe0f1e11ebb25e862ce3bc722af9a24425dec791b9d43cfbedd"),
    ("model_executor/models/glm4_moe.py",
     "7fa520422bdefc0751ba0175a0959300caad7128576de0802b5500cbc70d66e0"),
    ("model_executor/models/glm4_moe_mtp.py",
     "44d2e20dd4b27cb7e5d1b7f1a803b51e1b50be1fbeab92f09b4d62e0a2347472"),
    ("model_executor/models/deepseek_v2.py",
     "f9483594c8ccf5af3a67ab6eebd0d55ffee8db7fc79f9a0f30f6db8448dc67ed"),
    ("model_executor/layers/mla.py",
     "b509f0230356f851ef752070627382c541ba2757b7e9c6e68add3367162368b1"),
)
_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class VLLMReferenceStep:
    request_ids_before: tuple[str, ...]
    request_state_before_schedule: tuple[tuple[str, str, int], ...]
    request_state_after_schedule: tuple[tuple[str, str, int], ...]
    selected_request_ids: tuple[str, ...]
    scheduled_tokens: tuple[tuple[str, int], ...]
    computed_before_schedule: tuple[tuple[str, int], ...]
    computed_after_schedule: tuple[tuple[str, int], ...]
    computed_after_update: tuple[tuple[str, int | None], ...]
    spec_token_ids_before_schedule: tuple[
        tuple[str, tuple[int, ...]], ...
    ]
    scheduled_spec_decode_tokens: tuple[
        tuple[str, tuple[int, ...]], ...
    ]
    raw_sampled_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    mtp_drafter_executed: bool
    draft_token_ids_after_update: tuple[
        tuple[str, tuple[int, ...] | None], ...
    ]
    cached_tokens_after_schedule: tuple[tuple[str, int], ...]
    cached_request_new_token_ids: tuple[
        tuple[str, tuple[int, ...]], ...
    ]
    block_ids_before_schedule: tuple[
        tuple[str, tuple[tuple[int, ...], ...]], ...
    ]
    block_ids_after_schedule: tuple[
        tuple[str, tuple[tuple[int, ...], ...]], ...
    ]
    block_ids_after_schedule_all: tuple[
        tuple[str, tuple[tuple[int, ...], ...]], ...
    ]
    preempted_request_ids: tuple[str, ...]
    resumed_request_ids: tuple[str, ...]
    free_blocks_before_schedule: int
    free_blocks_after_schedule: int
    free_blocks_after_update: int
    generated_tokens: tuple[tuple[str, int], ...]
    finished_request_ids: tuple[str, ...]
    released_request_ids: tuple[str, ...]
    finish_reason_by_request: tuple[tuple[str, str | None], ...]
    block_ids_after_update: tuple[
        tuple[str, tuple[tuple[int, ...], ...] | None], ...
    ]


@dataclass(frozen=True)
class VLLMReferenceRun:
    version: str
    source_commit: str
    package_python_file_count: int
    package_python_manifest_sha256: str
    effective_chunked_prefill: bool
    effective_prefix_caching: bool
    tensor_parallel_size: int
    pipeline_parallel_size: int
    enable_expert_parallel: bool
    kv_cache_page_size_bytes: int
    kv_cache_tensor_owners: tuple[tuple[str, ...], ...]
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    kv_block_size_tokens: int
    num_kv_blocks: int
    scheduler_policy: str
    num_lookahead_slots: int
    speculative_method: str
    mtp_num_speculative_tokens: int
    mtp_fixed_accepted_tokens: int
    mtp_padded_drafter_batch: bool
    mtp_draft_model_type: str
    mtp_num_mtp_layers: int
    mtp_draft_max_model_len: int
    initial_free_blocks: int
    final_free_blocks: int
    final_unfinished_requests: int
    steps: tuple[VLLMReferenceStep, ...]


@dataclass(frozen=True)
class VLLMAdapterProbeStep:
    native: VLLMReferenceStep
    normalized_attention_kv_layout: str
    normalized_selected_request_ids: tuple[str, ...]
    normalized_scheduled_tokens: tuple[tuple[str, int], ...]
    normalized_computed_before: tuple[tuple[str, int], ...]
    normalized_recomputed_tokens: tuple[tuple[str, int], ...]
    normalized_prefix_replay_tokens: tuple[tuple[str, int], ...]
    normalized_prefix_hits: tuple[tuple[str, int, int, int], ...]
    normalized_resumed_request_ids: tuple[str, ...]
    normalized_preempted_request_ids: tuple[str, ...]
    normalized_generated_tokens: tuple[tuple[str, int], ...]
    normalized_mtp_rounds: tuple[
        tuple[str, int, int, int, int, int, int], ...
    ]
    normalized_finished_request_ids: tuple[str, ...]
    normalized_kv_cache_events: tuple[
        tuple[str, str, int, int, int], ...
    ]


@dataclass(frozen=True)
class VLLMAdapterProbeRun:
    version: str
    source_commit: str
    package_python_manifest_sha256: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    enable_expert_parallel: bool
    kv_cache_page_size_bytes: int
    kv_cache_tensor_owners: tuple[tuple[str, ...], ...]
    attention_kv_layout: str
    initial_free_blocks: int
    final_free_blocks: int
    final_unfinished_requests: int
    steps: tuple[VLLMAdapterProbeStep, ...]


@dataclass(frozen=True)
class _ReferenceModelView:
    """Independent minimal model surface consumed by upstream Scheduler."""

    max_model_len: int
    vocab_size: int
    dtype: Any
    num_hidden_layers: int
    num_kv_heads: int
    head_size: int
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    skip_tokenizer_init: bool = True

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_num_layers(self, parallel_config: Any) -> int:
        if parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "Kimi reference model view currently requires PP=1")
        return self.num_hidden_layers

    def get_num_kv_heads(self, parallel_config: Any) -> int:
        if parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "Kimi reference model view currently requires TP=1")
        return self.num_kv_heads

    def get_head_size(self) -> int:
        return self.head_size

    def get_layers_start_end_indices(
        self,
        parallel_config: Any,
    ) -> tuple[int, int]:
        return 0, self.get_num_layers(parallel_config)


@dataclass(frozen=True)
class _ReferenceVllmConfig:
    """Independent dependency bundle passed directly to vLLM Scheduler."""

    model_config: _ReferenceModelView
    cache_config: Any
    parallel_config: Any
    scheduler_config: Any
    structured_outputs_config: Any
    lora_config: None = None
    speculative_config: None = None
    kv_transfer_config: None = None
    kv_events_config: None = None


class _NVMLTripwire(ModuleType):
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"reference Scheduler attempted NVML access: {name}")


@dataclass(frozen=True)
class VLLMReferenceRequest:
    request_id: str
    input_tokens: int
    output_tokens: int
    admit_step: int = 0
    prefix_hit_tokens: int = 0
    initial_computed_tokens: int = 0
    initial_output_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("reference request_id must be non-empty")
        for name in (
                "input_tokens", "output_tokens", "admit_step",
                "prefix_hit_tokens", "initial_computed_tokens",
                "initial_output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"reference request {name} must be an integer")
        if min(self.input_tokens, self.output_tokens) < 1:
            raise ValueError(
                "reference input_tokens and output_tokens must be positive")
        if self.admit_step < 0:
            raise ValueError("reference admit_step cannot be negative")
        if not 0 <= self.prefix_hit_tokens <= self.input_tokens:
            raise ValueError(
                "reference prefix_hit_tokens must be between zero and input")
        if not 0 <= self.initial_computed_tokens <= self.input_tokens:
            raise ValueError(
                "reference initial_computed_tokens must be between zero and input")
        if not 0 <= self.initial_output_tokens < self.output_tokens:
            raise ValueError(
                "reference initial_output_tokens must be smaller than output")
        if self.initial_output_tokens:
            if self.initial_computed_tokens != self.input_tokens:
                raise ValueError(
                    "seeded reference output requires a fully computed prompt")
            if self.prefix_hit_tokens != 0:
                raise ValueError(
                    "seeded reference output cannot also inject a prefix hit")


@dataclass(frozen=True)
class VLLMReferenceMTPConfig:
    num_speculative_tokens: int = 0
    fixed_accepted_tokens: int = 0
    acceptance_source_id: str = "disabled"

    def __post_init__(self) -> None:
        for name in ("num_speculative_tokens", "fixed_accepted_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"reference MTP {name} must be an integer")
        if min(self.num_speculative_tokens, self.fixed_accepted_tokens) < 0:
            raise ValueError("reference MTP token counts cannot be negative")
        if self.fixed_accepted_tokens > self.num_speculative_tokens:
            raise ValueError(
                "reference MTP accepted tokens cannot exceed proposals")
        if (not isinstance(self.acceptance_source_id, str)
                or not self.acceptance_source_id):
            raise ValueError(
                "reference MTP acceptance_source_id must be non-empty")
        if self.num_speculative_tokens == 0:
            if self.fixed_accepted_tokens != 0:
                raise ValueError("disabled reference MTP cannot accept drafts")
            if self.acceptance_source_id != "disabled":
                raise ValueError(
                    "disabled reference MTP requires source ID 'disabled'")
        elif self.acceptance_source_id == "disabled":
            raise ValueError("enabled reference MTP requires provenance")


def _reference_scheduler_prefix_hit_tokens(
    request: VLLMReferenceRequest,
    block_size_tokens: int,
) -> int:
    """Mirror the pinned upstream full-block/last-token cache invariant."""

    reusable_tokens = min(
        request.prefix_hit_tokens,
        request.input_tokens - 1,
    )
    return reusable_tokens - reusable_tokens % block_size_tokens


def run_single_request_reference(
    *,
    model: Path,
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    max_batched_tokens: int,
    max_requests: int,
    num_kv_blocks: int,
    kv_block_size_tokens: int,
    max_model_len: int,
    dummy_token_id: int,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    mtp_config: VLLMReferenceMTPConfig | None = None,
) -> VLLMReferenceRun:
    """Run one request through an upstream Scheduler in a clean process."""

    return run_request_batch_reference(
        model=model,
        requests=(VLLMReferenceRequest(
            request_id,
            input_tokens,
            output_tokens,
        ),),
        max_batched_tokens=max_batched_tokens,
        max_requests=max_requests,
        num_kv_blocks=num_kv_blocks,
        kv_block_size_tokens=kv_block_size_tokens,
        max_model_len=max_model_len,
        dummy_token_id=dummy_token_id,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        mtp_config=mtp_config,
    )


def run_request_batch_reference(
    *,
    model: Path,
    requests: tuple[VLLMReferenceRequest, ...],
    max_batched_tokens: int,
    max_requests: int,
    num_kv_blocks: int,
    kv_block_size_tokens: int,
    max_model_len: int,
    dummy_token_id: int,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    mtp_config: VLLMReferenceMTPConfig | None = None,
) -> VLLMReferenceRun:
    """Run an explicit admission sequence through an upstream Scheduler."""

    if mtp_config is None:
        mtp_config = VLLMReferenceMTPConfig()
    elif not isinstance(mtp_config, VLLMReferenceMTPConfig):
        raise TypeError("mtp_config must be VLLMReferenceMTPConfig or null")
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_reference_worker,
        args=(
            child_connection,
            str(model.resolve()),
            requests,
            max_batched_tokens,
            max_requests,
            num_kv_blocks,
            kv_block_size_tokens,
            max_model_len,
            dummy_token_id,
            tensor_parallel_size,
            pipeline_parallel_size,
            enable_expert_parallel,
            mtp_config,
        ),
        name="midlayer-vllm-reference",
    )
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(_TIMEOUT_S):
            raise TimeoutError(
                f"vLLM reference did not respond within {_TIMEOUT_S}s")
        response = parent_connection.recv()
    finally:
        parent_connection.close()
        process.join(_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
            process.join(_TIMEOUT_S)
    if not isinstance(response, tuple) or not response:
        raise RuntimeError("vLLM reference returned an invalid response")
    if response[0] == "ok":
        if len(response) != 2 or not isinstance(response[1], VLLMReferenceRun):
            raise RuntimeError("vLLM reference returned malformed evidence")
        return response[1]
    if response[0] == "error" and len(response) == 4:
        raise RuntimeError(
            f"vLLM reference failed with {response[1]}: {response[2]}\n"
            f"{response[3]}")
    raise RuntimeError("vLLM reference returned an unknown response")


def run_single_request_adapter_probe(
    *,
    model: Path,
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    max_batched_tokens: int,
    num_kv_blocks: int,
    kv_block_size_tokens: int,
    max_model_len: int,
    dummy_token_id: int,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    mtp_config: VLLMReferenceMTPConfig | None = None,
    backend_role: str = "colocated",
    kv_bytes_per_token: int,
) -> VLLMAdapterProbeRun:
    """Record native state while driving the production control plane."""

    return run_request_batch_adapter_probe(
        model=model,
        requests=(VLLMReferenceRequest(
            request_id,
            input_tokens,
            output_tokens,
        ),),
        max_batched_tokens=max_batched_tokens,
        max_requests=1,
        num_kv_blocks=num_kv_blocks,
        kv_block_size_tokens=kv_block_size_tokens,
        max_model_len=max_model_len,
        dummy_token_id=dummy_token_id,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        mtp_config=mtp_config,
        backend_role=backend_role,
        kv_bytes_per_token=kv_bytes_per_token,
    )


def run_request_batch_adapter_probe(
    *,
    model: Path,
    requests: tuple[VLLMReferenceRequest, ...],
    max_batched_tokens: int,
    max_requests: int,
    num_kv_blocks: int,
    kv_block_size_tokens: int,
    max_model_len: int,
    dummy_token_id: int,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    mtp_config: VLLMReferenceMTPConfig | None = None,
    backend_role: str = "colocated",
    kv_bytes_per_token: int,
) -> VLLMAdapterProbeRun:
    """Record native state while driving multi-request production control."""

    if mtp_config is None:
        mtp_config = VLLMReferenceMTPConfig()
    elif not isinstance(mtp_config, VLLMReferenceMTPConfig):
        raise TypeError("mtp_config must be VLLMReferenceMTPConfig or null")
    if backend_role not in (
            "colocated", "prefill", "decode", "attention",
            "decode-attention"):
        raise ValueError(
            "adapter probe backend_role must be colocated, prefill, decode, "
            "attention, or decode-attention")
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_adapter_probe_worker,
        args=(
            child_connection,
            str(model.resolve()),
            requests,
            max_batched_tokens,
            max_requests,
            num_kv_blocks,
            kv_block_size_tokens,
            max_model_len,
            dummy_token_id,
            tensor_parallel_size,
            pipeline_parallel_size,
            enable_expert_parallel,
            mtp_config,
            backend_role,
            kv_bytes_per_token,
        ),
        name="midlayer-vllm-adapter-probe",
    )
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(_TIMEOUT_S):
            raise TimeoutError(
                f"vLLM adapter probe did not respond within {_TIMEOUT_S}s")
        response = parent_connection.recv()
    finally:
        parent_connection.close()
        process.join(_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
            process.join(_TIMEOUT_S)
    if not isinstance(response, tuple) or not response:
        raise RuntimeError("vLLM adapter probe returned an invalid response")
    if response[0] == "ok":
        if len(response) != 2 or not isinstance(
                response[1], VLLMAdapterProbeRun):
            raise RuntimeError("vLLM adapter probe returned malformed evidence")
        return response[1]
    if response[0] == "error" and len(response) == 4:
        raise RuntimeError(
            f"vLLM adapter probe failed with {response[1]}: {response[2]}\n"
            f"{response[3]}")
    raise RuntimeError("vLLM adapter probe returned an unknown response")


def _install_cpu_guard() -> Any:
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("importing PyTorch initialized a CUDA context")
    baseline_libraries = _loaded_accelerator_libraries()
    unexpected_baseline = tuple(
        path for path in sorted(baseline_libraries)
        if "libcuda.so" not in Path(path).name
    )
    if unexpected_baseline:
        raise RuntimeError(
            "PyTorch import loaded an unexpected NVIDIA library: "
            + ", ".join(unexpected_baseline))
    nvml = _NVMLTripwire("pynvml")
    nvml.__file__ = "<midlayer-reference-nvml-tripwire>"
    sys.modules["pynvml"] = nvml
    bundled_nvml = _NVMLTripwire("vllm.third_party.pynvml")
    bundled_nvml.__file__ = "<midlayer-reference-vllm-nvml-tripwire>"
    bundled_nvml.__package__ = "vllm.third_party"
    sys.modules["vllm.third_party.pynvml"] = bundled_nvml
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

    def forbidden(name: str):
        def blocked(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise RuntimeError(
                f"reference Scheduler attempted torch.cuda.{name}")
        return blocked

    for name in (
        "init",
        "_lazy_init",
        "current_device",
        "set_device",
        "synchronize",
        "get_device_capability",
        "get_device_name",
        "get_device_properties",
        "mem_get_info",
        "memory_allocated",
        "memory_reserved",
    ):
        if hasattr(torch.cuda, name):
            setattr(torch.cuda, name, forbidden(name))

    import vllm.platforms as platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms._current_platform = CpuPlatform()
    if not platforms.current_platform.is_cpu():
        raise RuntimeError("reference Scheduler did not select CpuPlatform")
    return torch, baseline_libraries


def _package_python_manifest(package_root: Path) -> tuple[int, str]:
    source_paths = sorted(
        package_root.rglob("*.py"),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    manifest = hashlib.sha256()
    for source_path in source_paths:
        relative_path = source_path.relative_to(package_root).as_posix()
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        manifest.update(relative_path.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(source_hash.encode("ascii"))
        manifest.update(b"\n")
    return len(source_paths), manifest.hexdigest()


def _verify_reference_source() -> tuple[str, str]:
    version = importlib.metadata.version("vllm")
    if version != _VLLM_VERSION:
        raise RuntimeError(f"expected vLLM {_VLLM_VERSION}, got {version}")
    package_spec = importlib.util.find_spec("vllm")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate vLLM for the reference Scheduler")
    locations = tuple(package_spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            f"reference vLLM must have one source location, got {locations!r}")
    package_root = Path(locations[0]).resolve()
    package_file_count, package_manifest = _package_python_manifest(
        package_root)
    if package_file_count != _PACKAGE_PYTHON_FILE_COUNT:
        raise RuntimeError(
            "reference vLLM Python source count differs from the audited "
            f"wheel: expected {_PACKAGE_PYTHON_FILE_COUNT}, "
            f"got {package_file_count}")
    if package_manifest != _PACKAGE_PYTHON_MANIFEST_SHA256:
        raise RuntimeError(
            "reference vLLM Python manifest differs from the audited wheel: "
            f"expected {_PACKAGE_PYTHON_MANIFEST_SHA256}, "
            f"got {package_manifest}")
    for relative_path, expected_hash in _SOURCE_HASHES:
        source_path = package_root / relative_path
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                "reference source differs from official commit "
                f"{_VLLM_SOURCE_COMMIT}: {relative_path} expected "
                f"{expected_hash}, got {actual_hash}")
    return version, package_manifest


def _loaded_accelerator_libraries() -> frozenset[str]:
    libraries: set[str] = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "libcuda.so" not in line and "libnvidia-ml.so" not in line:
            continue
        fields = line.split()
        if len(fields) < 6:
            raise RuntimeError(
                "cannot identify accelerator library mapping: " + line)
        libraries.add(fields[-1])
    return frozenset(libraries)


def _verify_no_device_access(
    torch: Any,
    baseline_libraries: frozenset[str],
) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("reference Scheduler initialized a CUDA context")
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            target = None
        if target is not None and (
                target.startswith("/dev/nvidia")
                or target.startswith("/dev/dri")):
            raise RuntimeError(
                "reference Scheduler opened an accelerator device: "
                f"{descriptor.name} -> {target}")
    unexpected_libraries = tuple(sorted(
        _loaded_accelerator_libraries() - baseline_libraries
    ))
    if unexpected_libraries:
        raise RuntimeError(
            "reference Scheduler loaded an NVIDIA library after the "
            "PyTorch import baseline: " + ", ".join(unexpected_libraries))


def _build_scheduler(
    *,
    model: str,
    max_batched_tokens: int,
    max_requests: int,
    num_kv_blocks: int,
    kv_block_size_tokens: int,
    max_model_len: int,
    torch: Any,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    enable_expert_parallel: bool,
    mtp_config: VLLMReferenceMTPConfig,
) -> tuple[Any, Any]:
    from vllm.config import (
        CacheConfig,
        ParallelConfig,
        SchedulerConfig,
        StructuredOutputsConfig,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MLAAttentionSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    if (pipeline_parallel_size != 1
            and "VLLM_PP_LAYER_PARTITION" in os.environ):
        raise NotImplementedError(
            "reference PP Scheduler requires the audited uniform layer "
            "partition and rejects VLLM_PP_LAYER_PARTITION")
    if mtp_config.num_speculative_tokens == 0:
        speculative_config = None
    else:
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": mtp_config.num_speculative_tokens,
        }
    raw_config = json.loads(
        (Path(model) / "config.json").read_text(encoding="utf-8"))
    is_kimi_mla = raw_config["model_type"] == "kimi_k25"
    if is_kimi_mla:
        provenance = json.loads(
            (Path(model) / "model_provenance.json").read_text(
                encoding="utf-8"))
        if (provenance["model_id"] != "moonshotai/Kimi-K2.6"
                or provenance["revision"] != (
                    "7eb5002f6aadc958aed6a9177b7ed26bb94011bb")):
            raise ValueError(
                "Kimi Scheduler reference requires the pinned Kimi-K2.6 "
                "revision")
        if (tensor_parallel_size != 1
                or pipeline_parallel_size != 1
                or enable_expert_parallel
                or mtp_config.num_speculative_tokens != 0):
            raise NotImplementedError(
                "independent Kimi MLA Scheduler reference requires "
                "TP=PP=1 with EP and MTP disabled")
        text_config = raw_config["text_config"]
        if text_config["model_type"] != "kimi_k2":
            raise ValueError("Kimi Scheduler reference requires kimi_k2 text")
        if text_config["architectures"] != ["DeepseekV3ForCausalLM"]:
            raise ValueError(
                "Kimi Scheduler reference requires DeepseekV3ForCausalLM")
        if (raw_config["dtype"] != "bfloat16"
                or text_config["dtype"] != "bfloat16"):
            raise ValueError("Kimi Scheduler reference requires BF16 compute")
        native_scheduler_config = SchedulerConfig(
            runner_type="generate",
            max_num_batched_tokens=max_batched_tokens,
            max_num_seqs=max_requests,
            max_model_len=max_model_len,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=0,
            enable_chunked_prefill=True,
            is_multimodal_model=False,
            policy="fcfs",
            num_lookahead_slots=0,
            disable_chunked_mm_input=False,
            disable_hybrid_kv_cache_manager=False,
            async_scheduling=False,
        )
        native_cache_config = CacheConfig(
            block_size=kv_block_size_tokens,
            cache_dtype="auto",
            is_attention_free=False,
            sliding_window=None,
            enable_prefix_caching=False,
        )
        native_parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
            enable_expert_parallel=False,
            enable_eplb=False,
            distributed_executor_backend="uni",
            decode_context_parallel_size=1,
        )
        model_view = _ReferenceModelView(
            max_model_len=max_model_len,
            vocab_size=text_config["vocab_size"],
            dtype=torch.bfloat16,
            num_hidden_layers=text_config["num_hidden_layers"],
            num_kv_heads=1,
            head_size=(
                text_config["kv_lora_rank"]
                + text_config["qk_rope_head_dim"]),
        )
        vllm_config = _ReferenceVllmConfig(
            model_config=model_view,
            cache_config=native_cache_config,
            parallel_config=native_parallel_config,
            scheduler_config=native_scheduler_config,
            structured_outputs_config=StructuredOutputsConfig(),
        )
    else:
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            dtype="bfloat16",
            max_model_len=max_model_len,
            enforce_eager=True,
            trust_remote_code=False,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            max_num_batched_tokens=max_batched_tokens,
            max_num_seqs=max_requests,
            block_size=kv_block_size_tokens,
            load_format="auto",
            skip_tokenizer_init=True,
            speculative_config=speculative_config,
        )
        vllm_config = engine_args.create_engine_config()
    if (vllm_config.parallel_config.tensor_parallel_size
            != tensor_parallel_size):
        raise RuntimeError(
            "reference vLLM rewrote tensor_parallel_size")
    if (vllm_config.parallel_config.pipeline_parallel_size
            != pipeline_parallel_size):
        raise RuntimeError(
            "reference vLLM rewrote pipeline_parallel_size")
    if (vllm_config.parallel_config.enable_expert_parallel
            is not enable_expert_parallel):
        raise RuntimeError(
            "reference vLLM rewrote enable_expert_parallel")
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    if scheduler_config.enable_chunked_prefill is not True:
        raise RuntimeError(
            "reference vLLM did not enable requested chunked prefill")
    if scheduler_config.chunked_prefill_enabled is not True:
        raise RuntimeError(
            "reference vLLM effective chunked prefill is not enabled")
    if scheduler_config.max_num_batched_tokens != max_batched_tokens:
        raise RuntimeError("reference vLLM rewrote max_num_batched_tokens")
    if scheduler_config.max_num_seqs != max_requests:
        raise RuntimeError("reference vLLM rewrote max_num_seqs")
    if scheduler_config.max_model_len != max_model_len:
        raise RuntimeError("reference vLLM rewrote max_model_len")
    if scheduler_config.policy != "fcfs":
        raise RuntimeError("reference vLLM scheduler policy is not fcfs")
    if (scheduler_config.num_lookahead_slots
            != mtp_config.num_speculative_tokens):
        raise RuntimeError("reference vLLM rewrote MTP lookahead")
    if mtp_config.num_speculative_tokens == 0:
        if vllm_config.speculative_config is not None:
            raise RuntimeError("reference vLLM enabled speculative decoding")
    else:
        speculative = vllm_config.speculative_config
        if speculative is None:
            raise RuntimeError("reference vLLM dropped MTP config")
        if speculative.method != "mtp":
            raise RuntimeError("reference vLLM rewrote MTP method")
        if (speculative.num_speculative_tokens
                != mtp_config.num_speculative_tokens):
            raise RuntimeError("reference vLLM rewrote MTP proposal count")
        if speculative.disable_padded_drafter_batch is not False:
            raise RuntimeError(
                "reference MTP requires pinned padded drafter batches")
        draft_model_config = speculative.draft_model_config
        if draft_model_config is None:
            raise RuntimeError("reference MTP draft model is missing")
        draft_hf_config = draft_model_config.hf_config
        if draft_hf_config.model_type != "glm4_moe_mtp":
            raise NotImplementedError(
                "reference workload supports only glm4_moe_mtp")
        if draft_hf_config.num_nextn_predict_layers != 1:
            raise NotImplementedError(
                "reference workload supports one GLM MTP layer")
        if draft_hf_config.num_hidden_layers != 0:
            raise RuntimeError(
                "reference GLM MTP draft must start at layer zero")
    if vllm_config.kv_transfer_config is not None:
        raise RuntimeError("reference vLLM enabled native KV transfer")
    if cache_config.enable_prefix_caching is not False:
        raise RuntimeError("reference vLLM enabled content prefix caching")
    if cache_config.block_size != kv_block_size_tokens:
        raise RuntimeError("reference vLLM rewrote KV block size")
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    if not isinstance(model_config.dtype, torch.dtype):
        raise TypeError("reference model dtype is not a torch.dtype")
    if is_kimi_mla:
        cache_spec = MLAAttentionSpec(
            block_size=kv_block_size_tokens,
            num_kv_heads=model_config.get_num_kv_heads(parallel_config),
            head_size=model_config.get_head_size(),
            dtype=model_config.dtype,
            cache_dtype_str=None,
        )
    else:
        cache_spec = FullAttentionSpec(
            block_size=kv_block_size_tokens,
            num_kv_heads=model_config.get_num_kv_heads(parallel_config),
            head_size=model_config.get_head_size(),
            dtype=model_config.dtype,
        )
    layer_start, layer_end = model_config.get_layers_start_end_indices(
        parallel_config)
    if layer_end - layer_start != model_config.get_num_layers(
            parallel_config):
        raise RuntimeError(
            "reference vLLM returned inconsistent PP layer boundaries")
    layer_names = [
        f"model.layers.{index}.self_attn.attn"
        for index in range(layer_start, layer_end)
    ]
    page_bytes = cache_spec.page_size_bytes
    kv_cache_config = KVCacheConfig(
        num_blocks=num_kv_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_kv_blocks * page_bytes,
                shared_by=[layer_name],
            )
            for layer_name in layer_names
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=cache_spec,
            )
        ],
    )
    actual_tensor_owners = tuple(
        tuple(tensor.shared_by)
        for tensor in kv_cache_config.kv_cache_tensors
    )
    expected_tensor_owners = tuple((layer_name,) for layer_name in layer_names)
    if actual_tensor_owners != expected_tensor_owners:
        raise RuntimeError(
            "reference KV tensors must be owned one per layer")
    vllm_config.cache_config.num_gpu_blocks = num_kv_blocks
    vllm_config.cache_config.num_gpu_blocks_override = num_kv_blocks
    scheduler = Scheduler(
        vllm_config,
        kv_cache_config,
        StructuredOutputManager(vllm_config),
    )
    return scheduler, vllm_config


def _block_ids(scheduler: Any, request_id: str) -> tuple[tuple[int, ...], ...]:
    groups = scheduler.kv_cache_manager.get_blocks(
        request_id).get_block_ids()
    return tuple(tuple(group) for group in groups)


def _verify_content_hash_state_disabled(native_request: Any) -> None:
    if native_request.block_hashes != []:
        raise RuntimeError(
            "length-only reference request unexpectedly contains block hashes")
    if native_request.get_hash_new_full_blocks is not None:
        raise RuntimeError(
            "length-only reference request unexpectedly enables content hashing")


def _reference_worker(
    connection: Connection,
    model: str,
    request_specs: tuple[VLLMReferenceRequest, ...],
    max_batched_tokens: int,
    max_requests: int,
    num_kv_blocks: int,
    kv_block_size_tokens: int,
    max_model_len: int,
    dummy_token_id: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    enable_expert_parallel: bool,
    mtp_config: VLLMReferenceMTPConfig,
) -> None:
    try:
        if "src.serving.backend.vllm" in sys.modules:
            raise RuntimeError("reference process imported the Midlayer adapter")
        version, package_manifest = _verify_reference_source()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = "1"
        torch, baseline_libraries = _install_cpu_guard()

        from vllm.sampling_params import SamplingParams
        from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
        from vllm.v1.request import Request

        scheduler, vllm_config = _build_scheduler(
            model=model,
            max_batched_tokens=max_batched_tokens,
            max_requests=max_requests,
            num_kv_blocks=num_kv_blocks,
            kv_block_size_tokens=kv_block_size_tokens,
            max_model_len=max_model_len,
            torch=torch,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            mtp_config=mtp_config,
        )
        if mtp_config.num_speculative_tokens:
            speculative_config = vllm_config.speculative_config
            if speculative_config is None:
                raise RuntimeError("reference MTP config disappeared")
            draft_model_config = speculative_config.draft_model_config
            if draft_model_config is None:
                raise RuntimeError("reference MTP draft model disappeared")
            mtp_draft_max_model_len = draft_model_config.max_model_len
            if mtp_draft_max_model_len is None:
                mtp_draft_max_model_len = (
                    vllm_config.model_config.max_model_len)
            if (isinstance(mtp_draft_max_model_len, bool)
                    or not isinstance(mtp_draft_max_model_len, int)
                    or mtp_draft_max_model_len <= 0):
                raise RuntimeError(
                    "reference MTP draft max model length must be positive")
        else:
            mtp_draft_max_model_len = 0
        if "src.serving.backend.vllm" in sys.modules:
            raise RuntimeError("reference construction imported the adapter")
        _verify_no_device_access(torch, baseline_libraries)
        vocab_size = vllm_config.model_config.get_vocab_size()
        if not 0 <= dummy_token_id < vocab_size:
            raise ValueError("reference dummy token is outside the vocabulary")
        if (not isinstance(request_specs, tuple) or not request_specs
                or any(not isinstance(spec, VLLMReferenceRequest)
                       for spec in request_specs)):
            raise TypeError(
                "reference requests must be a non-empty request tuple")
        request_ids = tuple(spec.request_id for spec in request_specs)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("reference request IDs must be unique")
        pending_specs = list(request_specs)
        block_pool = scheduler.kv_cache_manager.block_pool
        initial_free_blocks = block_pool.get_num_free_blocks()
        steps: list[VLLMReferenceStep] = []
        max_steps = sum(
            spec.input_tokens + spec.output_tokens + 1
            for spec in request_specs)
        while pending_specs or scheduler.has_unfinished_requests():
            if len(steps) >= max_steps:
                raise RuntimeError("reference Scheduler did not converge")
            step_index = len(steps)
            if any(spec.admit_step < step_index for spec in pending_specs):
                raise RuntimeError("reference request missed its admission step")
            admitting = tuple(
                spec for spec in pending_specs
                if spec.admit_step == step_index)
            pending_specs = [
                spec for spec in pending_specs
                if spec.admit_step != step_index
            ]
            for spec in admitting:
                native_request = Request(
                    request_id=spec.request_id,
                    prompt_token_ids=[dummy_token_id] * spec.input_tokens,
                    sampling_params=SamplingParams(
                        max_tokens=spec.output_tokens),
                    pooling_params=None,
                    eos_token_id=None,
                    arrival_time=float(spec.admit_step),
                    block_hasher=None,
                )
                if spec.initial_computed_tokens:
                    native_request.num_computed_tokens = (
                        spec.initial_computed_tokens)
                else:
                    native_request.num_computed_tokens = (
                        _reference_scheduler_prefix_hit_tokens(
                            spec, kv_block_size_tokens))
                for _ in range(spec.initial_output_tokens):
                    native_request.append_output_token_ids(dummy_token_id)
                _verify_content_hash_state_disabled(native_request)
                scheduler.add_request(native_request)
            if not scheduler.has_unfinished_requests():
                raise RuntimeError(
                    "reference admission plan has a step with no runnable work")
            selected_candidates = tuple(scheduler.requests)
            for native_id in selected_candidates:
                _verify_content_hash_state_disabled(
                    scheduler.requests[native_id])
            state_before_schedule = tuple(
                (
                    native_id,
                    scheduler.requests[native_id].status.name,
                    scheduler.requests[native_id].num_computed_tokens,
                )
                for native_id in selected_candidates
            )
            computed_before = tuple(
                (native_id, scheduler.requests[native_id].num_computed_tokens)
                for native_id in selected_candidates
            )
            blocks_before_schedule = tuple(
                (native_id, _block_ids(scheduler, native_id))
                for native_id in selected_candidates
            )
            spec_tokens_before_schedule = tuple(
                (
                    native_id,
                    tuple(scheduler.requests[native_id].spec_token_ids),
                )
                for native_id in selected_candidates
            )
            free_before = block_pool.get_num_free_blocks()
            scheduler_output = scheduler.schedule()
            state_after_schedule = tuple(
                (
                    native_id,
                    scheduler.requests[native_id].status.name,
                    scheduler.requests[native_id].num_computed_tokens,
                )
                for native_id in selected_candidates
            )
            state_before_by_id = {
                native_id: (status, computed)
                for native_id, status, computed in state_before_schedule
            }
            state_after_by_id = {
                native_id: (status, computed)
                for native_id, status, computed in state_after_schedule
            }
            preempted_ids = tuple(
                native_id
                for native_id in selected_candidates
                if state_before_by_id[native_id][0] != "PREEMPTED"
                and state_after_by_id[native_id][0] == "PREEMPTED"
            )
            cached_ids = tuple(
                scheduler_output.scheduled_cached_reqs.req_ids)
            resumed_flags = tuple(
                scheduler_output.scheduled_cached_reqs.
                resumed_from_preemption)
            if len(cached_ids) != len(resumed_flags):
                raise RuntimeError(
                    "reference Scheduler returned inconsistent resume flags")
            resumed_ids = tuple(
                native_id
                for native_id, resumed in zip(cached_ids, resumed_flags)
                if resumed
            )
            cached_new_token_ids = tuple(
                tuple(token_ids)
                for token_ids in (
                    scheduler_output.scheduled_cached_reqs.new_token_ids)
            )
            if cached_new_token_ids and (
                    len(cached_new_token_ids) != len(cached_ids)):
                raise RuntimeError(
                    "reference Scheduler returned inconsistent PP cached "
                    "token payloads")
            cached_token_payloads = tuple(zip(
                cached_ids,
                cached_new_token_ids,
            ))
            scheduled_tokens = tuple(
                (native_id, count)
                for native_id, count
                in scheduler_output.num_scheduled_tokens.items()
            )
            selected_ids = tuple(native_id for native_id, _ in scheduled_tokens)
            if not selected_ids:
                raise RuntimeError(
                    "reference Scheduler returned an empty nonterminal step")
            computed_after = tuple(
                (native_id, scheduler.requests[native_id].num_computed_tokens)
                for native_id in selected_ids
            )
            cached_tokens_after = tuple(
                (native_id, scheduler.requests[native_id].num_cached_tokens)
                for native_id in selected_ids
            )
            blocks_after_schedule = tuple(
                (native_id, _block_ids(scheduler, native_id))
                for native_id in selected_ids
            )
            blocks_after_schedule_all = tuple(
                (native_id, _block_ids(scheduler, native_id))
                for native_id in selected_candidates
            )
            free_after_schedule = block_pool.get_num_free_blocks()
            mtp_drafter_executed = bool(
                mtp_config.num_speculative_tokens
                and max(
                    scheduler.requests[native_id].num_computed_tokens
                    for native_id in selected_ids
                ) + mtp_config.num_speculative_tokens
                <= mtp_draft_max_model_len
            )

            sampled_token_ids: list[list[int]] = []
            for native_id in selected_ids:
                native_request = scheduler.requests[native_id]
                if native_id in scheduler_output.scheduled_spec_decode_tokens:
                    scheduled_drafts = list(
                        scheduler_output.scheduled_spec_decode_tokens[native_id])
                else:
                    scheduled_drafts = []
                if scheduled_drafts:
                    accepted = min(
                        mtp_config.fixed_accepted_tokens,
                        len(scheduled_drafts),
                    )
                    sampled = (
                        scheduled_drafts[:accepted] + [dummy_token_id])
                elif (native_request.num_computed_tokens
                        >= native_request.num_tokens):
                    sampled = [dummy_token_id]
                else:
                    sampled = []
                sampled_token_ids.append(sampled)
            model_output = ModelRunnerOutput(
                req_ids=list(selected_ids),
                req_id_to_index={
                    native_id: index
                    for index, native_id in enumerate(selected_ids)
                },
                sampled_token_ids=sampled_token_ids,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
            raw_sampled_token_ids = tuple(
                (native_id, tuple(sampled))
                for native_id, sampled in zip(
                    selected_ids,
                    sampled_token_ids,
                )
            )
            core_outputs = scheduler.update_from_output(
                scheduler_output, model_output)
            if mtp_drafter_executed:
                scheduler.update_draft_token_ids(DraftTokenIds(
                    req_ids=list(selected_ids),
                    draft_token_ids=[
                        [dummy_token_id] * mtp_config.num_speculative_tokens
                        for native_id in selected_ids
                    ],
                ))
            finished_ids = tuple(sorted(scheduler.finished_req_ids))
            finish_reasons: dict[str, str | None] = {
                native_id: None for native_id in selected_ids
            }
            emitted_tokens: dict[str, int] = {
                native_id: 0 for native_id in selected_ids
            }
            for client_index in sorted(core_outputs):
                for output in core_outputs[client_index].outputs:
                    if output.request_id not in finish_reasons:
                        raise RuntimeError(
                            "reference returned output for an unscheduled "
                            f"request: {output.request_id!r}")
                    finish_reasons[output.request_id] = (
                        None if output.finish_reason is None
                        else str(output.finish_reason)
                    )
                    emitted_tokens[output.request_id] += len(
                        output.new_token_ids)
            raw_counts = {
                native_id: len(sampled)
                for native_id, sampled in raw_sampled_token_ids
            }
            if any(
                    emitted_tokens[native_id] > raw_counts[native_id]
                    for native_id in selected_ids):
                raise RuntimeError(
                    "reference vLLM emitted more tokens than the raw model "
                    "output")
            released_ids = tuple(
                native_id for native_id in selected_ids
                if native_id not in scheduler.requests
            )
            for native_id in finished_ids:
                if native_id not in released_ids:
                    raise RuntimeError(
                        "finished reference request retained scheduler state: "
                        f"{native_id!r}")
                if finish_reasons[native_id] is None:
                    raise RuntimeError(
                        "finished reference request has no finish reason: "
                        f"{native_id!r}")
            blocks_after_update = tuple(
                (
                    native_id,
                    _block_ids(scheduler, native_id)
                    if native_id in scheduler.requests else None,
                )
                for native_id in selected_ids
            )
            computed_after_update = tuple(
                (
                    native_id,
                    scheduler.requests[native_id].num_computed_tokens
                    if native_id in scheduler.requests else None,
                )
                for native_id in selected_ids
            )
            drafts_after_update = tuple(
                (
                    native_id,
                    tuple(scheduler.requests[native_id].spec_token_ids)
                    if native_id in scheduler.requests else None,
                )
                for native_id in selected_ids
            )
            steps.append(VLLMReferenceStep(
                request_ids_before=selected_candidates,
                request_state_before_schedule=state_before_schedule,
                request_state_after_schedule=state_after_schedule,
                selected_request_ids=selected_ids,
                scheduled_tokens=scheduled_tokens,
                computed_before_schedule=tuple(
                    pair for pair in computed_before
                    if pair[0] in selected_ids
                ),
                computed_after_schedule=computed_after,
                computed_after_update=computed_after_update,
                spec_token_ids_before_schedule=spec_tokens_before_schedule,
                scheduled_spec_decode_tokens=tuple(
                    (
                        native_id,
                        tuple(
                            scheduler_output.scheduled_spec_decode_tokens[
                                native_id]
                        ),
                    )
                    for native_id in selected_ids
                    if native_id in (
                        scheduler_output.scheduled_spec_decode_tokens)
                ),
                raw_sampled_token_ids=raw_sampled_token_ids,
                mtp_drafter_executed=mtp_drafter_executed,
                draft_token_ids_after_update=drafts_after_update,
                cached_tokens_after_schedule=cached_tokens_after,
                cached_request_new_token_ids=cached_token_payloads,
                block_ids_before_schedule=blocks_before_schedule,
                block_ids_after_schedule=blocks_after_schedule,
                block_ids_after_schedule_all=blocks_after_schedule_all,
                preempted_request_ids=preempted_ids,
                resumed_request_ids=resumed_ids,
                free_blocks_before_schedule=free_before,
                free_blocks_after_schedule=free_after_schedule,
                free_blocks_after_update=block_pool.get_num_free_blocks(),
                generated_tokens=tuple(
                    (native_id, emitted_tokens[native_id])
                    for native_id in selected_ids
                ),
                finished_request_ids=finished_ids,
                released_request_ids=released_ids,
                finish_reason_by_request=tuple(
                    (native_id, finish_reasons[native_id])
                    for native_id in selected_ids
                ),
                block_ids_after_update=blocks_after_update,
            ))
            _verify_no_device_access(torch, baseline_libraries)

        result = VLLMReferenceRun(
            version=version,
            source_commit=_VLLM_SOURCE_COMMIT,
            package_python_file_count=_PACKAGE_PYTHON_FILE_COUNT,
            package_python_manifest_sha256=package_manifest,
            effective_chunked_prefill=(
                vllm_config.scheduler_config.chunked_prefill_enabled),
            effective_prefix_caching=(
                vllm_config.cache_config.enable_prefix_caching),
            tensor_parallel_size=(
                vllm_config.parallel_config.tensor_parallel_size),
            pipeline_parallel_size=(
                vllm_config.parallel_config.pipeline_parallel_size),
            enable_expert_parallel=(
                vllm_config.parallel_config.enable_expert_parallel),
            kv_cache_page_size_bytes=(
                scheduler.kv_cache_config.kv_cache_groups[0].
                kv_cache_spec.page_size_bytes),
            kv_cache_tensor_owners=tuple(
                tuple(tensor.shared_by)
                for tensor in scheduler.kv_cache_config.kv_cache_tensors
            ),
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens),
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_model_len=vllm_config.scheduler_config.max_model_len,
            kv_block_size_tokens=vllm_config.cache_config.block_size,
            num_kv_blocks=num_kv_blocks,
            scheduler_policy=vllm_config.scheduler_config.policy,
            num_lookahead_slots=(
                vllm_config.scheduler_config.num_lookahead_slots),
            speculative_method=(
                "mtp" if mtp_config.num_speculative_tokens else "disabled"),
            mtp_num_speculative_tokens=mtp_config.num_speculative_tokens,
            mtp_fixed_accepted_tokens=mtp_config.fixed_accepted_tokens,
            mtp_padded_drafter_batch=bool(
                mtp_config.num_speculative_tokens),
            mtp_draft_model_type=(
                "glm4_moe_mtp"
                if mtp_config.num_speculative_tokens else "disabled"),
            mtp_num_mtp_layers=(
                1 if mtp_config.num_speculative_tokens else 0),
            mtp_draft_max_model_len=mtp_draft_max_model_len,
            initial_free_blocks=initial_free_blocks,
            final_free_blocks=block_pool.get_num_free_blocks(),
            final_unfinished_requests=scheduler.get_num_unfinished_requests(),
            steps=tuple(steps),
        )
        connection.send(("ok", result))
    except BaseException as exc:
        connection.send((
            "error",
            type(exc).__name__,
            str(exc),
            traceback.format_exc(),
        ))
        raise
    finally:
        connection.close()


def _adapter_probe_worker(
    connection: Connection,
    model: str,
    request_specs: tuple[VLLMReferenceRequest, ...],
    max_batched_tokens: int,
    max_requests: int,
    num_kv_blocks: int,
    kv_block_size_tokens: int,
    max_model_len: int,
    dummy_token_id: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    enable_expert_parallel: bool,
    mtp_config: VLLMReferenceMTPConfig,
    backend_role: str,
    kv_bytes_per_token: int,
) -> None:
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = "1"

        from src.serving.backend.vllm_scheduler_process import (
            VLLM_SCHEDULER_SOURCE_COMMIT,
            _WORKER_ENVIRONMENT_FLAG,
            _install_cpu_device_guard,
            _verify_no_accelerator_access,
            _verify_pinned_vllm_source,
        )

        _, version, package_manifest = _verify_pinned_vllm_source()
        os.environ[_WORKER_ENVIRONMENT_FLAG] = "1"
        baseline_libraries = _install_cpu_device_guard()

        from src.serving.backend.vllm import (
            VLLMSchedulerConfig,
            _NativeVLLMControlPlane,
        )
        from src.serving.request_gen.request_stream import (
            RequestSpec,
            RequestStream,
            request_stream_sha256,
        )
        from src.serving.schedule.backend_protocol import BackendConfig

        backend_config = BackendConfig(
            "main",
            backend_role,
            max_batched_tokens=max_batched_tokens,
            max_requests=max_requests,
            num_kv_blocks=num_kv_blocks,
            kv_block_size_tokens=kv_block_size_tokens,
            kv_bytes_per_token=kv_bytes_per_token,
        scheduler_pipeline_depth=1)
        if (backend_role == "attention"
                and max_requests == 2
                and mtp_config.num_speculative_tokens != 0):
            bound_request_stream_sha256 = request_stream_sha256(RequestStream(
                RequestSpec(
                    float(spec.admit_step),
                    spec.request_id,
                    spec.input_tokens,
                    spec.output_tokens,
                    spec.prefix_hit_tokens,
                )
                for spec in request_specs
            ))
        else:
            bound_request_stream_sha256 = None
        scheduler_config = VLLMSchedulerConfig(
            model,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            dummy_token_id=dummy_token_id,
            mtp_num_speculative_tokens=mtp_config.num_speculative_tokens,
            mtp_acceptance_policy=(
                "fixed_count"
                if mtp_config.num_speculative_tokens else "disabled"),
            mtp_fixed_accepted_tokens=mtp_config.fixed_accepted_tokens,
            mtp_acceptance_source_id=mtp_config.acceptance_source_id,
            request_stream_sha256=bound_request_stream_sha256,
        )
        control_plane = _NativeVLLMControlPlane(
            backend_config, scheduler_config)
        captured_results: list[Any] = []
        captured_raw_model_outputs: list[
            tuple[tuple[str, tuple[int, ...]], ...]
        ] = []
        native_update = control_plane.scheduler.update_from_output

        def capture_update(scheduler_output: Any, model_output: Any) -> Any:
            captured_raw_model_outputs.append(tuple(
                (
                    request_id,
                    tuple(model_output.sampled_token_ids[
                        model_output.req_id_to_index[request_id]
                    ]),
                )
                for request_id in scheduler_output.num_scheduled_tokens
            ))
            result = native_update(scheduler_output, model_output)
            captured_results.append(result)
            return result

        control_plane.scheduler.update_from_output = capture_update
        if (not isinstance(request_specs, tuple) or not request_specs
                or any(not isinstance(spec, VLLMReferenceRequest)
                       for spec in request_specs)):
            raise TypeError(
                "adapter probe requests must be a non-empty request tuple")
        request_ids = tuple(spec.request_id for spec in request_specs)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("adapter probe request IDs must be unique")
        pending_specs = list(request_specs)
        block_pool = control_plane.scheduler.kv_cache_manager.block_pool
        initial_free_blocks = block_pool.get_num_free_blocks()
        probe_steps: list[VLLMAdapterProbeStep] = []
        max_steps = sum(
            spec.input_tokens + spec.output_tokens + 1
            for spec in request_specs)
        while pending_specs or control_plane.has_work():
            if len(probe_steps) >= max_steps:
                raise RuntimeError("adapter probe did not converge")
            step_index = len(probe_steps)
            if any(spec.admit_step < step_index for spec in pending_specs):
                raise RuntimeError("adapter probe request missed admission step")
            admitting = tuple(
                spec for spec in pending_specs
                if spec.admit_step == step_index)
            pending_specs = [
                spec for spec in pending_specs
                if spec.admit_step != step_index
            ]
            for spec in admitting:
                control_plane.add_request(RequestSpec(
                    float(spec.admit_step),
                    spec.request_id,
                    spec.input_tokens,
                    spec.output_tokens,
                    spec.prefix_hit_tokens,
                ))
            if not control_plane.has_work():
                raise RuntimeError(
                    "adapter probe admission plan has no runnable work")
            native_scheduler = control_plane.scheduler
            request_ids_before = tuple(native_scheduler.requests)
            for native_id in request_ids_before:
                _verify_content_hash_state_disabled(
                    native_scheduler.requests[native_id])
            state_before_schedule = tuple(
                (
                    native_id,
                    native_scheduler.requests[native_id].status.name,
                    native_scheduler.requests[native_id].num_computed_tokens,
                )
                for native_id in request_ids_before
            )
            computed_before = tuple(
                (
                    native_id,
                    native_scheduler.requests[native_id].num_computed_tokens,
                )
                for native_id in request_ids_before
            )
            blocks_before = tuple(
                (native_id, _block_ids(native_scheduler, native_id))
                for native_id in request_ids_before
            )
            spec_tokens_before_schedule = tuple(
                (
                    native_id,
                    tuple(native_scheduler.requests[native_id].spec_token_ids),
                )
                for native_id in request_ids_before
            )
            free_before = block_pool.get_num_free_blocks()

            normalized_step = control_plane.plan_step(float(len(probe_steps)))
            if normalized_step is None:
                raise RuntimeError("adapter probe returned no step with work")
            selected_ids = tuple(
                request.request_id
                for request in normalized_step.schedule.requests
            )
            proposer_depths = {
                request.mtp_proposer_tokens
                for request in normalized_step.schedule.requests
            }
            if proposer_depths == {0}:
                mtp_drafter_executed = False
            elif proposer_depths == {mtp_config.num_speculative_tokens}:
                mtp_drafter_executed = True
            else:
                raise RuntimeError(
                    "adapter returned inconsistent padded proposer depths")
            scheduled_tokens = tuple(
                (request.request_id, request.scheduled_tokens)
                for request in normalized_step.schedule.requests
            )
            state_after_schedule = tuple(
                (
                    native_id,
                    native_scheduler.requests[native_id].status.name,
                    native_scheduler.requests[native_id].num_computed_tokens,
                )
                for native_id in request_ids_before
            )
            state_before_by_id = {
                native_id: (status, computed)
                for native_id, status, computed in state_before_schedule
            }
            state_after_by_id = {
                native_id: (status, computed)
                for native_id, status, computed in state_after_schedule
            }
            preempted_ids = tuple(
                native_id
                for native_id in request_ids_before
                if state_before_by_id[native_id][0] != "PREEMPTED"
                and state_after_by_id[native_id][0] == "PREEMPTED"
            )
            if control_plane._pending is None:
                raise RuntimeError(
                    "adapter probe has no native pending SchedulerOutput")
            native_scheduler_output = control_plane._pending[0][1]
            cached_ids = tuple(
                native_scheduler_output.scheduled_cached_reqs.req_ids)
            resumed_flags = tuple(
                native_scheduler_output.scheduled_cached_reqs.
                resumed_from_preemption)
            if len(cached_ids) != len(resumed_flags):
                raise RuntimeError(
                    "adapter probe captured inconsistent resume flags")
            resumed_ids = tuple(
                native_id
                for native_id, resumed in zip(cached_ids, resumed_flags)
                if resumed
            )
            cached_new_token_ids = tuple(
                tuple(token_ids)
                for token_ids in (
                    native_scheduler_output.scheduled_cached_reqs.
                    new_token_ids)
            )
            if cached_new_token_ids and (
                    len(cached_new_token_ids) != len(cached_ids)):
                raise RuntimeError(
                    "adapter probe captured inconsistent PP cached token "
                    "payloads")
            cached_token_payloads = tuple(zip(
                cached_ids,
                cached_new_token_ids,
            ))
            computed_after = tuple(
                (
                    native_id,
                    native_scheduler.requests[native_id].num_computed_tokens,
                )
                for native_id in selected_ids
            )
            cached_tokens_after = tuple(
                (
                    native_id,
                    native_scheduler.requests[native_id].num_cached_tokens,
                )
                for native_id in selected_ids
            )
            blocks_after_schedule = tuple(
                (native_id, _block_ids(native_scheduler, native_id))
                for native_id in selected_ids
            )
            blocks_after_schedule_all = tuple(
                (native_id, _block_ids(native_scheduler, native_id))
                for native_id in request_ids_before
            )
            free_after_schedule = block_pool.get_num_free_blocks()

            control_plane.complete_step(
                normalized_step,
                float(len(probe_steps) + 1),
                execute_model=False,
                execute_transfers=False,
                sample_tokens=False,
            )
            if len(captured_results) != len(probe_steps) + 1:
                raise RuntimeError(
                    "adapter probe did not capture exactly one native update")
            if len(captured_raw_model_outputs) != len(probe_steps) + 1:
                raise RuntimeError(
                    "adapter probe did not capture exactly one model output")
            core_outputs = captured_results[-1]
            raw_model_output = captured_raw_model_outputs[-1]
            finish_reasons: dict[str, str | None] = {
                native_id: None for native_id in selected_ids
            }
            generated_counts: dict[str, int] = {
                native_id: 0 for native_id in selected_ids
            }
            for client_index in sorted(core_outputs):
                for output in core_outputs[client_index].outputs:
                    if output.request_id not in finish_reasons:
                        raise RuntimeError(
                            "adapter probe received output for an unscheduled "
                            f"request: {output.request_id!r}")
                    finish_reasons[output.request_id] = (
                        None if output.finish_reason is None
                        else str(output.finish_reason)
                    )
                    generated_counts[output.request_id] += len(
                        output.new_token_ids)
            finished_ids = tuple(sorted(native_scheduler.finished_req_ids))
            released_ids = tuple(
                native_id for native_id in selected_ids
                if native_id not in native_scheduler.requests
            )
            blocks_after_update = tuple(
                (
                    native_id,
                    _block_ids(native_scheduler, native_id)
                    if native_id in native_scheduler.requests else None,
                )
                for native_id in selected_ids
            )
            computed_after_update = tuple(
                (
                    native_id,
                    native_scheduler.requests[native_id].num_computed_tokens
                    if native_id in native_scheduler.requests else None,
                )
                for native_id in selected_ids
            )
            drafts_after_update = tuple(
                (
                    native_id,
                    tuple(native_scheduler.requests[native_id].spec_token_ids)
                    if native_id in native_scheduler.requests else None,
                )
                for native_id in selected_ids
            )
            native_evidence = VLLMReferenceStep(
                request_ids_before=request_ids_before,
                request_state_before_schedule=state_before_schedule,
                request_state_after_schedule=state_after_schedule,
                selected_request_ids=selected_ids,
                scheduled_tokens=scheduled_tokens,
                computed_before_schedule=tuple(
                    pair for pair in computed_before
                    if pair[0] in selected_ids
                ),
                computed_after_schedule=computed_after,
                computed_after_update=computed_after_update,
                spec_token_ids_before_schedule=spec_tokens_before_schedule,
                scheduled_spec_decode_tokens=tuple(
                    (
                        native_id,
                        tuple(
                            native_scheduler_output.scheduled_spec_decode_tokens[
                                native_id]
                        ),
                    )
                    for native_id in selected_ids
                    if native_id in (
                        native_scheduler_output.scheduled_spec_decode_tokens)
                ),
                raw_sampled_token_ids=raw_model_output,
                mtp_drafter_executed=mtp_drafter_executed,
                draft_token_ids_after_update=drafts_after_update,
                cached_tokens_after_schedule=cached_tokens_after,
                cached_request_new_token_ids=cached_token_payloads,
                block_ids_before_schedule=blocks_before,
                block_ids_after_schedule=blocks_after_schedule,
                block_ids_after_schedule_all=blocks_after_schedule_all,
                preempted_request_ids=preempted_ids,
                resumed_request_ids=resumed_ids,
                free_blocks_before_schedule=free_before,
                free_blocks_after_schedule=free_after_schedule,
                free_blocks_after_update=block_pool.get_num_free_blocks(),
                generated_tokens=tuple(
                    (native_id, generated_counts[native_id])
                    for native_id in selected_ids
                ),
                finished_request_ids=finished_ids,
                released_request_ids=released_ids,
                finish_reason_by_request=tuple(
                    (native_id, finish_reasons[native_id])
                    for native_id in selected_ids
                ),
                block_ids_after_update=blocks_after_update,
            )
            probe_steps.append(VLLMAdapterProbeStep(
                native=native_evidence,
                normalized_attention_kv_layout=(
                    normalized_step.schedule.attention_kv_layout),
                normalized_selected_request_ids=selected_ids,
                normalized_scheduled_tokens=scheduled_tokens,
                normalized_computed_before=tuple(
                    (
                        request.request_id,
                        request.computed_tokens,
                    )
                    for request in normalized_step.schedule.requests
                ),
                normalized_recomputed_tokens=tuple(
                    (
                        request.request_id,
                        request.recomputed_tokens,
                    )
                    for request in normalized_step.schedule.requests
                ),
                normalized_prefix_replay_tokens=tuple(
                    (
                        request.request_id,
                        request.prefix_replay_tokens,
                    )
                    for request in normalized_step.schedule.requests
                ),
                normalized_prefix_hits=tuple(
                    (
                        request.request_id,
                        request.trace_prefix_hit_tokens,
                        request.scheduler_prefix_hit_tokens,
                        request.prefix_scheduler_delta_tokens,
                    )
                    for request in normalized_step.schedule.requests
                ),
                normalized_resumed_request_ids=tuple(
                    request.request_id
                    for request in normalized_step.schedule.requests
                    if request.resumed_from_preemption
                ),
                normalized_preempted_request_ids=(
                    normalized_step.schedule.preempted_request_ids),
                normalized_generated_tokens=tuple(sorted(
                    normalized_step.generated_tokens.items()
                )),
                normalized_mtp_rounds=tuple(
                    (
                        request.request_id,
                        request.mtp_proposed_tokens,
                        request.mtp_verified_tokens,
                        request.mtp_accepted_tokens,
                        request.mtp_rejected_tokens,
                        request.mtp_proposer_tokens,
                        request.mtp_next_proposed_tokens,
                    )
                    for request in normalized_step.schedule.requests
                ),
                normalized_finished_request_ids=(
                    normalized_step.finished_request_ids),
                normalized_kv_cache_events=tuple(
                    (
                        event.request_id,
                        event.action.value,
                        event.token_start,
                        event.token_end,
                        event.bytes,
                    )
                    for event in normalized_step.kv_cache_events
                ),
            ))
            _verify_no_accelerator_access(baseline_libraries)

        result = VLLMAdapterProbeRun(
            version=version,
            source_commit=VLLM_SCHEDULER_SOURCE_COMMIT,
            package_python_manifest_sha256=package_manifest,
            tensor_parallel_size=(
                control_plane.vllm_config.parallel_config.
                tensor_parallel_size),
            pipeline_parallel_size=(
                control_plane.vllm_config.parallel_config.
                pipeline_parallel_size),
            enable_expert_parallel=(
                control_plane.vllm_config.parallel_config.
                enable_expert_parallel),
            kv_cache_page_size_bytes=(
                control_plane.scheduler.kv_cache_config.kv_cache_groups[0].
                kv_cache_spec.page_size_bytes),
            kv_cache_tensor_owners=tuple(
                tuple(tensor.shared_by)
                for tensor in (
                    control_plane.scheduler.kv_cache_config.kv_cache_tensors)
            ),
            attention_kv_layout=control_plane.attention_kv_layout,
            initial_free_blocks=initial_free_blocks,
            final_free_blocks=block_pool.get_num_free_blocks(),
            final_unfinished_requests=(
                control_plane.scheduler.get_num_unfinished_requests()),
            steps=tuple(probe_steps),
        )
        connection.send(("ok", result))
    except BaseException as exc:
        connection.send((
            "error",
            type(exc).__name__,
            str(exc),
            traceback.format_exc(),
        ))
        raise
    finally:
        connection.close()
