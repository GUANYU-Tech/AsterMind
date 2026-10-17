"""Pinned vLLM KV allocator bridge for analytical Serving capacity."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import Connection
import os
import traceback

from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.workload.model_config import ModelConfig


_ALLOCATOR_TIMEOUT_S = 60.0
_ALLOCATOR_SYMBOL = "vllm.v1.core.kv_cache_utils.get_kv_cache_configs"
_SCHEDULER_CONFIG_SYMBOL = (
    "vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config")


def _pinned_allocator_identity() -> dict[str, str]:
    # Keep this import lazy: importing the backend package while the KV-cache
    # package is initializing would create a package-level dependency cycle.
    from src.serving.backend.vllm_scheduler_process import (
        VLLM_SCHEDULER_SOURCE_COMMIT,
        VLLM_SCHEDULER_VERSION,
        vllm_scheduler_source_identity,
    )

    source_identity = vllm_scheduler_source_identity()
    return {
        "version": VLLM_SCHEDULER_VERSION,
        "source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
        "package_python_manifest_sha256": source_identity[
            "package_python_manifest_sha256"],
        "allocator_symbol": _ALLOCATOR_SYMBOL,
        "scheduler_config_symbol": _SCHEDULER_CONFIG_SYMBOL,
    }


@dataclass(frozen=True)
class VLLMKVAllocation:
    """Serializable facts returned by vLLM's native KV allocator."""

    version: str
    source_commit: str
    package_python_manifest_sha256: str
    allocator_symbol: str
    scheduler_config_symbol: str
    kv_spec_type: str
    kv_block_size_tokens: int
    max_model_len: int
    kv_page_size_bytes_per_layer: int
    num_kv_blocks: int
    worker_num_kv_blocks: tuple[int, ...]
    worker_allocated_tensor_bytes: tuple[int, ...]
    scheduler_tensor_owners: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        expected_identity = _pinned_allocator_identity()
        for name, expected in expected_identity.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"native vLLM KV allocation {name} disagrees with the "
                    "pinned allocator")
        if self.kv_spec_type not in (
                "FullAttentionSpec", "MLAAttentionSpec",
                "HeterogeneousMLAAttentionSpec"):
            raise ValueError(
                "native vLLM KV allocation has an unsupported KV spec type")
        for name in (
                "kv_block_size_tokens", "max_model_len",
                "kv_page_size_bytes_per_layer", "num_kv_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"native vLLM KV allocation {name} must be a positive "
                    "integer")
        if (not isinstance(self.worker_num_kv_blocks, tuple)
                or not self.worker_num_kv_blocks
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value != self.num_kv_blocks
                       for value in self.worker_num_kv_blocks)):
            raise ValueError(
                "native vLLM KV allocation must report the final block count "
                "for every worker")
        if (not isinstance(self.worker_allocated_tensor_bytes, tuple)
                or len(self.worker_allocated_tensor_bytes)
                != len(self.worker_num_kv_blocks)
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1
                       for value in self.worker_allocated_tensor_bytes)):
            raise ValueError(
                "native vLLM KV allocation must report positive allocated "
                "tensor bytes for every worker")
        if (not isinstance(self.scheduler_tensor_owners, tuple)
                or not self.scheduler_tensor_owners
                or any(not isinstance(owners, tuple) or not owners
                       or any(not isinstance(owner, str) or not owner
                              for owner in owners)
                       or len(owners) != len(set(owners))
                       for owners in self.scheduler_tensor_owners)):
            raise ValueError(
                "native vLLM KV allocation Scheduler tensor owners must be "
                "non-empty unique layer names")


@dataclass
class _AllocatorModelView:
    max_model_len: int


@dataclass(frozen=True)
class _AllocatorCacheView:
    num_gpu_blocks_override: None = None


@dataclass(frozen=True)
class _AllocatorSchedulerView:
    disable_hybrid_kv_cache_manager: bool = False


@dataclass(frozen=True)
class _AllocatorParallelView:
    decode_context_parallel_size: int = 1


@dataclass(frozen=True)
class _AllocatorVllmConfig:
    model_config: _AllocatorModelView
    cache_config: _AllocatorCacheView = _AllocatorCacheView()
    scheduler_config: _AllocatorSchedulerView = _AllocatorSchedulerView()
    parallel_config: _AllocatorParallelView = _AllocatorParallelView()


def _native_allocate(
    model: ModelConfig,
    parallelism: ParallelConfig,
    kv_block_size_tokens: int,
    max_model_len: int,
    available_memory_by_rank: tuple[int, ...],
) -> VLLMKVAllocation:
    from src.serving.backend.vllm_scheduler_process import (
        VLLM_SCHEDULER_SOURCE_COMMIT,
        _install_cpu_device_guard,
        _verify_no_accelerator_access,
        _verify_pinned_vllm_source,
    )

    _, version, package_manifest = _verify_pinned_vllm_source()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_CPU_KVCACHE_SPACE"] = "1"
    baseline_libraries = _install_cpu_device_guard()

    import torch
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MLAAttentionSpec,
    )

    if model.compute_dtype == "bf16":
        dtype = torch.bfloat16
    elif model.compute_dtype == "fp16":
        dtype = torch.float16
    else:
        raise NotImplementedError(
            "pinned vLLM KV allocation supports BF16 or FP16 model dtype")
    tp_size = parallelism.tensor_parallel_size
    pp_size = parallelism.pipeline_parallel_size
    layer_ranges = pipeline_layer_ranges_for_model(model, pp_size)
    if model.attention_layout == "standard_gqa":
        spec = FullAttentionSpec(
            block_size=kv_block_size_tokens,
            num_kv_heads=model.kv_heads_per_tensor_parallel_rank(tp_size),
            head_size=model.head_dim,
            dtype=dtype,
        )
        spec_type = "FullAttentionSpec"
    elif model.attention_layout == "mla":
        if model.compute_dtype not in ("bf16", "fp16"):
            raise NotImplementedError(
                "MLA native KV allocation requires BF16 or FP16 cache dtype")
        spec = MLAAttentionSpec(
            block_size=kv_block_size_tokens,
            num_kv_heads=1,
            head_size=model.kv_lora_rank + model.qk_rope_head_dim,
            dtype=dtype,
            cache_dtype_str=None,
        )
        spec_type = "MLAAttentionSpec"
    elif model.attention_layout == "dsa_mla":
        if model.index_head_dim % 128:
            raise ValueError(
                "DSA index head dimension must divide its 128-element scale "
                "group")
        spec = MLAAttentionSpec(
            block_size=kv_block_size_tokens,
            num_kv_heads=1,
            head_size=model.kv_lora_rank + model.qk_rope_head_dim,
            dtype=dtype,
            cache_dtype_str=None,
        )
        index_spec = MLAAttentionSpec(
            block_size=kv_block_size_tokens,
            num_kv_heads=1,
            head_size=(
                model.index_head_dim + model.index_head_dim // 128 * 4),
            dtype=torch.uint8,
            cache_dtype_str="fp8",
        )
        spec_type = "HeterogeneousMLAAttentionSpec"
    else:
        raise AssertionError(
            f"unhandled attention layout: {model.attention_layout!r}")

    specs_by_rank = []
    for rank in range(parallelism.worker_count):
        pp_rank = (rank // tp_size) % pp_size
        layer_start, layer_end = layer_ranges[pp_rank]
        rank_specs = {
            f"model.layers.{layer_index}.self_attn.attn": spec
            for layer_index in range(layer_start, layer_end)
        }
        if model.attention_layout == "dsa_mla":
            rank_specs.update({
                f"model.layers.{layer_index}.self_attn.indexer.k_cache": (
                    index_spec)
                for layer_index in range(layer_start, layer_end)
                if model.indexer_types[layer_index] == "full"
            })
        specs_by_rank.append(rank_specs)
    vllm_config = _AllocatorVllmConfig(
        model_config=_AllocatorModelView(max_model_len=max_model_len))
    if model.attention_layout == "dsa_mla":
        # vLLM 0.11.0's multi-worker helper keeps one global
        # UniformTypeKVCacheSpecs payload when PP workers have heterogeneous
        # MLA/index-cache tensors.  That makes every worker reserve the whole
        # model's tensor set.  Invoke the same native allocator once per worker
        # and then apply its documented global-min block reconciliation.
        configs = [
            get_kv_cache_configs(
                vllm_config,
                [rank_specs],
                [available_memory],
            )[0]
            for rank_specs, available_memory in zip(
                specs_by_rank, available_memory_by_rank, strict=True)
        ]
        min_num_blocks = min(config.num_blocks for config in configs)
        for config in configs:
            config.num_blocks = min_num_blocks
    elif model.attention_layout in ("standard_gqa", "mla"):
        configs = get_kv_cache_configs(
            vllm_config,
            specs_by_rank,
            list(available_memory_by_rank),
        )
    else:
        raise AssertionError(
            "unhandled allocator attention layout: "
            f"{model.attention_layout!r}")
    if len(configs) != len(specs_by_rank):
        raise RuntimeError(
            "pinned vLLM allocator returned the wrong worker count")
    worker_blocks = tuple(config.num_blocks for config in configs)
    if not worker_blocks or len(set(worker_blocks)) != 1:
        raise RuntimeError(
            "pinned vLLM allocator did not converge to one Scheduler block "
            f"count: {worker_blocks!r}")
    scheduler_config = generate_scheduler_kv_cache_config(configs)
    if scheduler_config.num_blocks != worker_blocks[0]:
        raise RuntimeError(
            "pinned vLLM Scheduler KV config changed the final block count")
    _verify_no_accelerator_access(baseline_libraries)
    return VLLMKVAllocation(
        version=version,
        source_commit=VLLM_SCHEDULER_SOURCE_COMMIT,
        package_python_manifest_sha256=package_manifest,
        allocator_symbol=_ALLOCATOR_SYMBOL,
        scheduler_config_symbol=_SCHEDULER_CONFIG_SYMBOL,
        kv_spec_type=spec_type,
        kv_block_size_tokens=kv_block_size_tokens,
        max_model_len=max_model_len,
        kv_page_size_bytes_per_layer=spec.page_size_bytes,
        num_kv_blocks=worker_blocks[0],
        worker_num_kv_blocks=worker_blocks,
        worker_allocated_tensor_bytes=tuple(
            sum(tensor.size for tensor in config.kv_cache_tensors)
            for config in configs
        ),
        scheduler_tensor_owners=tuple(
            tuple(tensor.shared_by)
            for tensor in scheduler_config.kv_cache_tensors
        ),
    )


def _allocator_worker(
    connection: Connection,
    model: ModelConfig,
    parallelism: ParallelConfig,
    kv_block_size_tokens: int,
    max_model_len: int,
    available_memory_by_rank: tuple[int, ...],
) -> None:
    try:
        result = _native_allocate(
            model,
            parallelism,
            kv_block_size_tokens,
            max_model_len,
            available_memory_by_rank,
        )
        connection.send(("ok", result))
    except BaseException as error:
        connection.send((
            "error",
            type(error).__name__,
            str(error),
            traceback.format_exc(),
        ))
        raise
    finally:
        connection.close()


def allocate_vllm_kv_cache(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    kv_block_size_tokens: int,
    max_model_len: int,
    available_memory_by_rank: tuple[int, ...],
) -> VLLMKVAllocation:
    """Run the pinned native allocator without initializing an accelerator."""

    if not isinstance(model, ModelConfig):
        raise TypeError("vLLM KV allocation requires ModelConfig")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("vLLM KV allocation requires ParallelConfig")
    for name, value in (
            ("kv_block_size_tokens", kv_block_size_tokens),
            ("max_model_len", max_model_len)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (not isinstance(available_memory_by_rank, tuple)
            or len(available_memory_by_rank) != parallelism.worker_count
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in available_memory_by_rank)):
        raise ValueError(
            "available_memory_by_rank must contain one positive integer for "
            "every physical worker rank")

    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_allocator_worker,
        args=(
            child_connection,
            model,
            parallelism,
            kv_block_size_tokens,
            max_model_len,
            available_memory_by_rank,
        ),
        name="midlayer-vllm-kv-allocator",
    )
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(_ALLOCATOR_TIMEOUT_S):
            process.terminate()
            process.join(5.0)
            raise TimeoutError(
                "pinned vLLM KV allocator did not respond within 60 seconds")
        response = parent_connection.recv()
    finally:
        parent_connection.close()
        process.join(5.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
    if (not isinstance(response, tuple) or not response):
        raise RuntimeError("pinned vLLM KV allocator returned an invalid response")
    if response[0] == "ok":
        if len(response) != 2 or not isinstance(response[1], VLLMKVAllocation):
            raise RuntimeError(
                "pinned vLLM KV allocator returned an invalid allocation")
        return response[1]
    if response[0] == "error" and len(response) == 4:
        raise RuntimeError(
            "pinned vLLM KV allocator failed with "
            f"{response[1]}: {response[2]}\n{response[3]}")
    raise RuntimeError("pinned vLLM KV allocator returned an unknown response")
