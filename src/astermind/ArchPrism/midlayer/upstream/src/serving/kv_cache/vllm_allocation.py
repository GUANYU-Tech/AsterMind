"""Pinned vLLM KV allocator bridge for analytical Serving capacity."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import traceback

from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.workload.model_config import (
    ModelConfig, is_validated_qwen3_moe_ep2_model_scope, is_validated_qwen3_pipeline_model_scope,
)


_ALLOCATOR_TIMEOUT_S = 60.0
_ALLOCATOR_SYMBOL = "vllm.v1.core.kv_cache_utils.get_kv_cache_configs"
_SCHEDULER_CONFIG_SYMBOL = (
    "vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config")


def _vllm030_full_gqa_model(model):
    """Qwen3MoeAttention uses the same ordinary Attention KV owner as dense Qwen.

    FFN experts do not create attention cache owners. Quantized weights and
    other attention families remain outside this source projection.
    """
    return (model.model_type in ("qwen3", "qwen3_moe", "llama")
        and model.attention_layout == "standard_gqa" and model.weight_dtype == "bf16"
        and ((model.model_type == "qwen3_moe" and model.num_experts > 0)
             or (model.model_type != "qwen3_moe" and model.num_experts == 0)))


def _vllm030_kimi_mla_model(model):
    return (model.model_id == "moonshotai/Kimi-K2.6" and model.model_type == "kimi_k2"
        and model.attention_layout == "mla" and model.compute_dtype == "bf16" and model.weight_dtype == "mixed_int4_bf16")


def vllm030_supports_expert_parallel_cache(model, tensor_parallel_size):
    """Cache placement domains, independent of executing expert workers.

    GLM/K2.6 DP/CP1 TP=EP8 source partitions FFN experts, not its
    replicated latent/indexer attention caches. It has the same per-layer MLA spec
    as EP-off. Actual expert/communication providers are not established.
    """
    if not isinstance(model, ModelConfig) or type(tensor_parallel_size) is not int:
        raise TypeError("0.30 expert cache scope requires ModelConfig and integer TP")
    glm = (model.model_type == "glm_moe_dsa" and model.attention_layout == "dsa_mla"
        and model.model_id in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3")
        and model.compute_dtype == "bf16" and model.weight_dtype in ("bf16", "fp8"))
    return ((tensor_parallel_size == 2 and is_validated_qwen3_moe_ep2_model_scope(model))
        or (tensor_parallel_size == 8 and (_vllm030_kimi_mla_model(model) or glm)))


def vllm030_uses_plain_mla_cache(model):
    """Ordinary latent/RoPE owners shared by V3 and the Kimi text wrapper."""
    return (_vllm030_kimi_mla_model(model)
        or (model.model_id == "deepseek-ai/DeepSeek-V3" and model.model_type == "deepseek_v3"
            and model.num_hidden_layers == 61 and model.attention_layout == "mla"
            and model.compute_dtype == "bf16" and model.weight_dtype == "fp8"))


def _pinned_allocator_identity(version: str | None = None) -> dict[str, str]:
    # Keep this import lazy: importing the backend package while the KV-cache
    # package is initializing would create a package-level dependency cycle.
    from src.serving.backend.vllm_scheduler_process import (
        VLLM_SCHEDULER_VERSION,
        vllm030_allocator_source_identity,
        vllm_scheduler_source_identity,
    )

    if version is None or version == VLLM_SCHEDULER_VERSION:
        source_identity = vllm_scheduler_source_identity()
    elif version == "0.30.0":
        source_identity = vllm030_allocator_source_identity()
    else:
        raise ValueError("native vLLM KV allocation version disagrees with supported pinned allocators")
    return {
        "version": source_identity["version"],
        "source_commit": source_identity["source_commit"],
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
        expected_identity = _pinned_allocator_identity(self.version)
        for name, expected in expected_identity.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"native vLLM KV allocation {name} disagrees with the "
                    "pinned allocator")
        supported_specs = (("UniformTypeKVCacheSpecs", "FullAttentionSpec", "MLAAttentionSpec") if self.version == "0.30.0" else
                           ("FullAttentionSpec", "MLAAttentionSpec", "HeterogeneousMLAAttentionSpec"))
        if self.kv_spec_type not in supported_specs:
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


def validate_vllm030_cache_placement(vllm_config, worker, *, available_memory: int | None = None) -> int:
    """Validate ordinary 0.30 shared-arena descriptors using upstream planning.

    Source-verified CPU callers supply genuine native config objects. Re-plan
    metadata only; no cache tensor/device allocation is performed. The return
    value is the backing size, not proof of a normal available-KV budget.
    """
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupRole, KVCacheGroupSpec, KVCacheTensor

    if not isinstance(vllm_config, VllmConfig) or not isinstance(worker, KVCacheConfig):
        raise TypeError("0.30 cache placement requires genuine native configurations")
    if (vllm_config.device_config.device_type != "cpu"
            or vllm_config.attention_config.hisparse_config is not None):
        raise NotImplementedError("0.30 placement validation requires ordinary guarded CPU configs")
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        raise ValueError("0.30 placement validation rejects debug block overrides")
    if type(worker.num_blocks) is not int or worker.num_blocks <= 0:
        raise ValueError("0.30 cache placement requires a positive native block count")
    if (not isinstance(worker.kv_cache_groups, list) or not worker.kv_cache_groups
            or any(not isinstance(group, KVCacheGroupSpec) or group.host_resident is not False
                   or group.is_eagle_group is not False or group.role is not KVCacheGroupRole.DEFAULT
                   for group in worker.kv_cache_groups)):
        raise ValueError("0.30 cache placement requires ordinary device-only target groups")
    if (not isinstance(worker.kv_cache_tensors, list) or not worker.kv_cache_tensors
            or any(not isinstance(tensor, KVCacheTensor) for tensor in worker.kv_cache_tensors)):
        raise ValueError("0.30 cache placement requires native tensor descriptors")
    for tensor in worker.kv_cache_tensors:
        if (any(type(getattr(tensor, name)) is not int for name in
                ("size", "offset", "layer_stride", "block_stride"))
                or tensor.size <= 0 or tensor.offset < 0 or tensor.layer_stride <= 0
                or tensor.block_stride <= 0 or tensor.host_resident is not False):
            raise ValueError("0.30 cache placement has invalid arena size, offset or strides")
    sizes = {tensor.size for tensor in worker.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("0.30 cache placement descriptors must share one backing size")
    arena_bytes = next(iter(sizes))
    if available_memory is not None and (type(available_memory) is not int or available_memory <= 0
                                         or arena_bytes > available_memory):
        raise ValueError("0.30 cache placement exceeds the declared byte budget")
    planned = get_kv_cache_config_from_groups(vllm_config, worker.kv_cache_groups, arena_bytes)
    if planned != worker:
        raise ValueError("0.30 cache placement disagrees with source-planned offsets, strides, owners or block count")
    return arena_bytes


def allocate_vllm030_glm_cache(
    vllm_config,
    model: ModelConfig,
    parallelism: ParallelConfig,
    available_memory_by_rank: tuple[int, ...],
    *,
    source_root: Path,
    supported_layouts_by_rank: tuple[tuple[str, ...], ...],
):
    """Compatibility entry for the existing ordinary GLM source allocation."""
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("0.30 GLM allocation requires ModelConfig and ParallelConfig")
    if (model.model_type != "glm_moe_dsa"
            or model.model_id not in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3")
            or model.compute_dtype != "bf16"):
        raise NotImplementedError("0.30 cache allocation requires an accurate BF16-compute GLM config")
    return allocate_vllm030_model_cache(vllm_config, model, parallelism, available_memory_by_rank,
        source_root=source_root, supported_layouts_by_rank=supported_layouts_by_rank)


def allocate_vllm030_model_cache(
    vllm_config,
    model: ModelConfig,
    parallelism: ParallelConfig,
    available_memory_by_rank: tuple[int, ...],
    *,
    source_root: Path,
    supported_layouts_by_rank: tuple[tuple[str, ...], ...],
):
    """Project target-off cache specs and call the actual 0.30 allocator.

    Runs inside a guarded, source-verified CPU process with a real VllmConfig.
    Input byte budgets and layout capabilities are caller declarations,
    not proof of normal H200 residency/reserves or worker observations. The model
    layers and distributed workers are not instantiated here. No debug block
    override or 0.11 allocation adapter is used. Returned upstream objects
    preserve group roles, strides and offsets.
    """
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("0.30 model allocation requires ModelConfig and ParallelConfig")
    full_gqa = _vllm030_full_gqa_model(model)
    if ((model.model_id not in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3",
                               "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro")
            and not full_gqa and not vllm030_uses_plain_mla_cache(model))
            or model.compute_dtype != "bf16"):
        raise NotImplementedError("0.30 cache allocation requires a supported BF16-compute source model")
    if (not parallelism.is_rank_local or (parallelism.pipeline_parallel_size != 1 and not (
            not parallelism.enable_expert_parallel
            and is_validated_qwen3_pipeline_model_scope(model, parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size)))
            or (parallelism.enable_expert_parallel and not vllm030_supports_expert_parallel_cache(
                model, parallelism.tensor_parallel_size))):
        raise NotImplementedError("0.30 model cache allocation requires rank-local supported PP and expert cache placement")
    if (not isinstance(available_memory_by_rank, tuple)
            or len(available_memory_by_rank) != parallelism.worker_count
            or any(type(value) is not int or value <= 0 for value in available_memory_by_rank)):
        raise ValueError("0.30 allocator requires one explicit positive byte budget per rank")
    if (not isinstance(supported_layouts_by_rank, tuple)
            or len(supported_layouts_by_rank) != parallelism.worker_count
            or any(not isinstance(layouts, tuple) or not layouts
                   or any(not isinstance(name, str) or not name for name in layouts)
                   for layouts in supported_layouts_by_rank)):
        raise ValueError("0.30 allocation requires explicit layout capabilities for every rank")
    if (any(layouts != supported_layouts_by_rank[0] for layouts in supported_layouts_by_rank)
            or any(len(set(layouts)) != len(layouts) for layouts in supported_layouts_by_rank)):
        raise ValueError("0.30 rank layout capabilities must agree and contain no duplicates")
    from src.serving.backend.vllm_scheduler_process import vllm030_source_identity
    vllm030_source_identity(source_root)
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config, get_kv_cache_configs,
    )
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.kv_cache_layout import KVCacheLayout

    if not isinstance(vllm_config, VllmConfig):
        raise TypeError("0.30 allocator requires the real upstream VllmConfig")
    if any(name not in KVCacheLayout.__members__ for name in supported_layouts_by_rank[0]):
        raise ValueError("0.30 allocation requires canonical native layout names")
    if (vllm_config.cache_config.kv_cache_layout is not None
            and vllm_config.cache_config.get_resolved_kv_cache_layout().name
            not in supported_layouts_by_rank[0]):
        raise ValueError("0.30 resolved layout conflicts with declared rank capabilities")
    if vllm_config.device_config.device_type != "cpu":
        raise NotImplementedError("0.30 control-plane allocation requires a CPU process")
    native_parallel = vllm_config.parallel_config
    if (native_parallel.tensor_parallel_size != parallelism.tensor_parallel_size
            or native_parallel.pipeline_parallel_size != parallelism.pipeline_parallel_size
            or native_parallel.decode_context_parallel_size != 1
            or native_parallel.prefill_context_parallel_size != 1
            or native_parallel.data_parallel_size != 1
            or native_parallel.enable_expert_parallel != parallelism.enable_expert_parallel):
        raise ValueError("0.30 config parallelism disagrees with the declared cache ranks")
    if vllm_config.speculative_config is not None:
        validate_vllm030_glm_mtp_config(vllm_config, model)
    if vllm_config.attention_config.hisparse_config is not None:
        raise NotImplementedError("0.30 HiSparse cache placement is not bound")
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        raise ValueError("0.30 byte-budget allocation rejects a debug block override")
    if model.model_type == "glm_moe_dsa" and vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("0.30 GLM source cache path requires BF16 MLA states")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("0.30 native model dtype disagrees with BF16 compute source")
    specs = vllm030_model_cache_specs(vllm_config, model)
    worker_specs = vllm030_worker_cache_specs(vllm_config, model)
    resolve_kv_cache_layout(
        vllm_config, [list(layouts) for layouts in supported_layouts_by_rank], specs.values())
    native_configs = get_kv_cache_configs(
        vllm_config, list(worker_specs),
        list(available_memory_by_rank))
    if len(native_configs) != parallelism.worker_count:
        raise RuntimeError("0.30 allocator returned an inconsistent worker count")
    for config, budget, expected in zip(native_configs, available_memory_by_rank, worker_specs, strict=True):
        owners = [name for group in config.kv_cache_groups for name in group.layer_names]
        if len(owners) != len(expected) or set(owners) != set(expected):
            raise RuntimeError("0.30 native cache grouping lost or duplicated source cache owners")
        validate_vllm030_cache_placement(vllm_config, config, available_memory=budget)
    scheduler_config = generate_scheduler_kv_cache_config(native_configs)
    if any(config.num_blocks != scheduler_config.num_blocks for config in native_configs):
        raise RuntimeError("0.30 allocator did not reconcile worker block counts")
    return native_configs, scheduler_config


def vllm030_v4_cache_specs(vllm_config, model: ModelConfig):
    """FlashMLA/H200 source cache map for Flash TP4 or Pro TP8, target-off.

    Construct native 0.30 spec objects, not model layers or GPU buffers.
    SWA and FP32 compressor states are sliding groups; compressed KV and
    index caches retain their distinct tokens_per_state and page padding.
    These specs do not establish a normal available-memory budget.
    """
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec, get_kv_quant_mode

    if not isinstance(vllm_config, VllmConfig) or not isinstance(model, ModelConfig):
        raise TypeError("0.30 V4 cache map requires genuine VllmConfig and ModelConfig")
    if (model.model_id not in ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro")
            or model.model_type != "deepseek_v4"):
        raise NotImplementedError("0.30 V4 cache map supports accurate V4 Flash/Pro only")
    parallel = vllm_config.parallel_config
    if ((model.model_id, parallel.tensor_parallel_size) not in (
                ("deepseek-ai/DeepSeek-V4-Flash", 4), ("deepseek-ai/DeepSeek-V4-Pro", 8))
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1 or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1 or parallel.enable_expert_parallel
            or vllm_config.speculative_config is not None
            or vllm_config.attention_config.hisparse_config is not None):
        raise NotImplementedError("0.30 V4 cache map requires Flash/TP4 or Pro/TP8, PP1/CP1/DP1/EP-off/MTP-off")
    if (vllm_config.model_config.dtype != torch.bfloat16
            or vllm_config.cache_config.cache_dtype != "fp8_ds_mla"):
        raise NotImplementedError("0.30 V4 FlashMLA source map requires BF16 compute and fp8_ds_mla cache")
    hf = vllm_config.model_config.hf_text_config
    for name, expected in (("num_hidden_layers", model.num_hidden_layers), ("head_dim", model.head_dim),
                           ("index_head_dim", model.index_head_dim), ("sliding_window", model.window_size)):
        if getattr(hf, name) != expected:
            raise ValueError(f"0.30 V4 source config disagrees on {name}")
    if tuple(hf.compress_ratios) != model.compress_ratios:
        raise ValueError("0.30 V4 source config disagrees on compress_ratios")
    block = vllm_config.cache_config.block_size
    if type(block) is not int or block <= 0:
        raise ValueError("0.30 V4 compressed-cache block size must be explicit and positive")
    specs = {}
    # Official config also carries a draft-tail entry. Target-off constructs
    # exactly num_hidden_layers modules, never a cache for that extra entry.
    for layer in range(model.num_hidden_layers):
        raw_ratio = model.compress_ratios[layer]
        ratio = max(1, raw_ratio)
        prefix = f"model.layers.{layer}.attn"
        specs[prefix + ".swa_cache"] = SlidingWindowMLASpec(
            block_size=64, num_kv_heads=1, head_size=model.head_dim, dtype=torch.uint8,
            sliding_window=model.window_size, cache_dtype_str="fp8_ds_mla", state_content_bytes=584,
            alignment=576, model_version="deepseek_v4", kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"))
        if ratio == 1:
            continue
        if ratio not in (4, 128):
            raise NotImplementedError("0.30 V4 source cache map requires C4/C128 compressors")
        specs[prefix] = MLAAttentionSpec(
            block_size=block, num_kv_heads=1, head_size=model.head_dim, dtype=torch.uint8,
            tokens_per_state=ratio, cache_dtype_str="fp8_ds_mla", alignment=576,
            model_version="deepseek_v4", kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"), state_content_bytes=584)
        factor = 2 if ratio == 4 else 1
        specs[prefix + ".compressor.state_cache"] = SlidingWindowMLASpec(
            block_size=4 if ratio == 4 else 8, num_kv_heads=1,
            head_size=2 * factor * model.head_dim, dtype=torch.float32,
            sliding_window=factor * ratio, alignment=576)
        if ratio == 4:
            specs[prefix + ".indexer.k_cache"] = MLAAttentionSpec(
                block_size=block, num_kv_heads=1, head_size=model.index_head_dim + model.index_head_dim // 128 * 4,
                dtype=torch.uint8, tokens_per_state=ratio, alignment=576)
            specs[prefix + ".indexer.compressor.state_cache"] = SlidingWindowMLASpec(
                block_size=4, num_kv_heads=1, head_size=4 * model.index_head_dim,
                dtype=torch.float32, sliding_window=8, alignment=576)
    return specs


def normalize_vllm030_glm_allocation(vllm_config, model: ModelConfig, workers: tuple,
                                   scheduler_cache, *, source_root: Path) -> VLLMKVAllocation:
    """Retain the ordinary GLM-only allocation normalization entry."""
    if not isinstance(model, ModelConfig) or model.model_type != "glm_moe_dsa":
        raise NotImplementedError("0.30 GLM allocation normalization requires GLM")
    result = normalize_vllm030_model_allocation(vllm_config, model, workers, scheduler_cache, source_root=source_root)
    if any(len(worker.kv_cache_groups) != 1 for worker in workers):
        raise NotImplementedError("0.30 GLM facts require the ordinary single uniform-type source group")
    return result


def normalize_vllm030_model_allocation(vllm_config, model: ModelConfig, workers: tuple,
                                     scheduler_cache, *, source_root: Path) -> VLLMKVAllocation:
    """Normalize actual GLM/V4 native objects into the existing facts.

    Shared-arena bytes are counted once per worker. These facts do not supply
    weight/reserve provenance, normal available-KV or model/provider evidence.
    The legacy scalar page field describes the first leaf only; complete
    mixed-group pool geometry must come from the validated worker objects.
    """
    from src.serving.backend.vllm_scheduler_process import vllm030_allocator_source_identity
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, KVCacheConfig, UniformTypeKVCacheSpecs

    if not isinstance(model, ModelConfig):
        raise TypeError("0.30 allocation facts require ModelConfig")
    identity = vllm030_allocator_source_identity(source_root)
    rank_specs = vllm030_worker_cache_specs(vllm_config, model)
    if (not isinstance(workers, tuple) or len(workers) != len(rank_specs)
            or any(not isinstance(worker, KVCacheConfig) for worker in workers)
            or not isinstance(scheduler_cache, KVCacheConfig)):
        raise TypeError("0.30 allocation facts require all genuine worker/Scheduler configurations")
    arenas = []
    for rank, (worker, expected_specs) in enumerate(zip(workers, rank_specs, strict=True)):
        arenas.append(validate_vllm030_cache_placement(vllm_config, worker))
        if worker.num_blocks != workers[0].num_blocks:
            raise ValueError("0.30 allocation facts disagree on reconciled rank block counts")
        tp = vllm_config.parallel_config.tensor_parallel_size
        if worker.kv_cache_groups != workers[rank // tp * tp].kv_cache_groups:
            raise ValueError("0.30 cache groups disagree within a tensor-parallel stage")
        actual_specs = {}
        for group in worker.kv_cache_groups:
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                members = group.kv_cache_spec.kv_cache_specs
            elif _vllm030_full_gqa_model(model) and type(group.kv_cache_spec) is FullAttentionSpec:
                members = {name: group.kv_cache_spec for name in group.layer_names}
            elif vllm030_uses_plain_mla_cache(model) and type(group.kv_cache_spec) is MLAAttentionSpec:
                members = {name: group.kv_cache_spec for name in group.layer_names}
            else:
                raise NotImplementedError("0.30 allocation facts require supported source group types")
            if (len(group.layer_names) != len(members) or set(group.layer_names) != set(members)
                    or set(actual_specs).intersection(members)):
                raise ValueError("0.30 allocation facts have inconsistent or duplicate group owners")
            actual_specs.update(members)
        if actual_specs != expected_specs:
            raise ValueError("0.30 allocation facts disagree with complete source cache owners")
    if scheduler_cache != generate_scheduler_kv_cache_config(list(workers)):
        raise ValueError("0.30 allocation facts disagree with the actual Scheduler representative")
    first_group = workers[0].kv_cache_groups[0].kv_cache_spec
    if isinstance(first_group, UniformTypeKVCacheSpecs):
        first_spec, spec_type = first_group.first_spec, "UniformTypeKVCacheSpecs"
    elif _vllm030_full_gqa_model(model) and type(first_group) is FullAttentionSpec:
        first_spec, spec_type = first_group, "FullAttentionSpec"
    elif vllm030_uses_plain_mla_cache(model) and type(first_group) is MLAAttentionSpec:
        first_spec, spec_type = first_group, "MLAAttentionSpec"
    else:
        raise NotImplementedError("0.30 source allocation has an unsupported group type")
    return VLLMKVAllocation(
        **identity, allocator_symbol=_ALLOCATOR_SYMBOL, scheduler_config_symbol=_SCHEDULER_CONFIG_SYMBOL,
        kv_spec_type=spec_type, kv_block_size_tokens=vllm_config.cache_config.block_size,
        max_model_len=vllm_config.model_config.max_model_len,
        kv_page_size_bytes_per_layer=first_spec.page_size_bytes,
        num_kv_blocks=scheduler_cache.num_blocks, worker_num_kv_blocks=tuple(worker.num_blocks for worker in workers),
        worker_allocated_tensor_bytes=tuple(arenas),
        scheduler_tensor_owners=tuple(tuple(tensor.layers) for tensor in scheduler_cache.kv_cache_tensors))


def vllm030_model_cache_specs(vllm_config, model: ModelConfig):
    """One source-map dispatch shared by allocator, normalization and backend."""
    if model.model_type == "glm_moe_dsa":
        return vllm030_glm_cache_specs(vllm_config, model)
    if model.model_type == "deepseek_v4":
        return vllm030_v4_cache_specs(vllm_config, model)
    if vllm030_uses_plain_mla_cache(model):
        return _vllm030_plain_mla_cache_specs(vllm_config, model)
    if _vllm030_full_gqa_model(model):
        return vllm030_dense_cache_specs(vllm_config, model)
    raise NotImplementedError("0.30 allocator has no source cache map for this model type")


def vllm030_worker_cache_specs(vllm_config, model: ModelConfig):
    """Project each worker's source layer ownership in PP-major/TP-minor order."""
    specs = vllm030_model_cache_specs(vllm_config, model)
    parallel = vllm_config.parallel_config
    if parallel.pipeline_parallel_size == 1:
        return tuple(dict(specs) for _ in range(parallel.tensor_parallel_size))
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
    # Each stage owns its local decoder layers; each TP lane uses the
    # source-sharded KV head descriptors for those layers.
    ranges = pipeline_layer_ranges_for_model(model, parallel.pipeline_parallel_size)
    return tuple({f"model.layers.{layer}.self_attn.attn": specs[f"model.layers.{layer}.self_attn.attn"]
        for layer in range(first, end)} for first, end in ranges for _ in range(parallel.tensor_parallel_size))


def vllm030_dense_cache_specs(vllm_config, model: ModelConfig):
    """Full GQA source descriptors, with no tensor allocation.

    Qwen3Attention and Qwen3MoeAttention construct one Attention per decoder. Its
    get_kv_cache_spec uses FullAttentionSpec for non-sliding decoder attention.
    This projects that constructor, not an instantiated GPU attention backend.
    """
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    parallel = vllm_config.parallel_config
    if (not _vllm030_full_gqa_model(model) or model.compute_dtype != "bf16"):
        raise NotImplementedError("dense cache source requires BF16 Qwen3/Llama full GQA PP1")
    if parallel.pipeline_parallel_size != 1 and (parallel.enable_expert_parallel
            or not is_validated_qwen3_pipeline_model_scope(model, parallel.tensor_parallel_size, parallel.pipeline_parallel_size)):
        raise NotImplementedError("0.30 pipeline cache source requires uniform untied Qwen3 TP1 PP or TP2/PP2, EP-off")
    hf = vllm_config.model_config.hf_config.to_dict()
    if model.model_type == "qwen3" and (hf["rms_norm_eps"] != model.rms_norm_eps
            or hf["hidden_act"] != model.hidden_act):
        raise ValueError("native Qwen3 optional configuration disagrees with source-normalized metadata")
    if (vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16")
            or ("sliding_window" in hf and hf["sliding_window"] is not None)
            or ("use_sliding_window" in hf and hf["use_sliding_window"] is not False)
            or ("layer_types" in hf and hf["layer_types"] is not None
                and any(layer != "full_attention" for layer in hf["layer_types"]))):
        raise NotImplementedError("dense cache source excludes quantized and sliding-window cache")
    block = vllm_config.cache_config.block_size
    if type(block) is not int or block <= 0:
        raise ValueError("dense Qwen cache source requires an explicit positive block size")
    return {f"model.layers.{layer}.self_attn.attn": FullAttentionSpec(
        block_size=block, num_kv_heads=model.kv_heads_per_tensor_parallel_rank(
            vllm_config.parallel_config.tensor_parallel_size),
        head_size=model.head_dim, head_size_v=model.head_dim, dtype=torch.bfloat16)
        for layer in range(model.num_hidden_layers)}


def vllm030_kimi_cache_specs(vllm_config, model: ModelConfig):
    """Compatibility entry retaining the exact K2.6 checkpoint boundary."""
    if not isinstance(model, ModelConfig):
        raise TypeError("0.30 Kimi cache projection requires ModelConfig")
    if not _vllm030_kimi_mla_model(model):
        raise NotImplementedError("0.30 Kimi cache projection requires its exact INT4/BF16 checkpoint")
    return _vllm030_plain_mla_cache_specs(vllm_config, model)


def _vllm030_plain_mla_cache_specs(vllm_config, model: ModelConfig):
    """DeepseekV2MLA/MLAAttention cache roles for V3 and Kimi's text wrapper.

    The ordinary BF16 CP1 branch has one replicated latent/RoPE cache per
    decoder layer. Weight quantization does not change its storage dtype.
    """
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    if not isinstance(vllm_config, VllmConfig) or not isinstance(model, ModelConfig):
        raise TypeError("0.30 Kimi cache projection requires real VllmConfig and ModelConfig")
    if not vllm030_uses_plain_mla_cache(model):
        raise NotImplementedError("0.30 ordinary MLA cache projection requires its supported checkpoint")
    parallel = vllm_config.parallel_config
    if (parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1
            or (parallel.enable_expert_parallel and not vllm030_supports_expert_parallel_cache(
                model, parallel.tensor_parallel_size)) or vllm_config.speculative_config is not None
            or vllm_config.attention_config.hisparse_config is not None
            or vllm_config.cache_config.sliding_window is not None
            or not vllm_config.model_config.use_mla):
        raise NotImplementedError("0.30 Kimi cache projection requires ordinary MLA PP1/CP1/DP1/MTP-off, EP-off or exact K2.6 EP8")
    if (vllm_config.model_config.dtype != torch.bfloat16
            or vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16")):
        raise NotImplementedError("0.30 Kimi cache projection requires BF16 latent storage")
    hf = vllm_config.model_config.hf_text_config
    for name, expected in (("hidden_size", model.hidden_size), ("num_hidden_layers", model.num_hidden_layers),
            ("kv_lora_rank", model.kv_lora_rank), ("qk_rope_head_dim", model.qk_rope_head_dim)):
        if getattr(hf, name) != expected:
            raise ValueError(f"0.30 Kimi config disagrees with source cache field {name}")
    block = vllm_config.cache_config.block_size
    if type(block) is not int or block <= 0:
        raise ValueError("0.30 Kimi cache block size must be explicit and positive")
    # mla_attention.py::get_kv_cache_spec, no indexer, window or draft.
    prefix = "language_model.model" if _vllm030_kimi_mla_model(model) else "model"
    return {f"{prefix}.layers.{layer}.self_attn.attn": MLAAttentionSpec(
        block_size=block, num_kv_heads=1, head_size=model.kv_lora_rank + model.qk_rope_head_dim,
        dtype=torch.bfloat16, cache_dtype_str=vllm_config.cache_config.cache_dtype)
        for layer in range(model.num_hidden_layers)}


def validate_vllm030_glm_mtp_config(vllm_config, model: ModelConfig):
    """Validate the source's exact GLM sequential, padded depth-two config.

    This checks native metadata only; it does not instantiate a draft model,
    allocate device tensors, validate a provider or derive a memory budget.
    """
    spec = vllm_config.speculative_config
    if (model.model_id not in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3")
            or model.mtp_architecture != "glm_moe_dsa_mtp" or model.num_mtp_layers != 1
            or spec is None or spec.method != "mtp" or spec.num_speculative_tokens != 2
            or spec.parallel_drafting or spec.disable_padded_drafter_batch
            or spec.use_local_argmax_reduction or spec.draft_sample_method != "greedy"
            or vllm_config.parallel_config.tensor_parallel_size not in (1, 8)):
        raise NotImplementedError("0.30 MTP placement requires accurate GLM sequential padded depth2 TP1/TP8 full-logits source")
    draft = spec.draft_model_config
    if draft is None:
        raise ValueError("0.30 GLM MTP requires a native draft model config")
    hf = draft.hf_config
    if (hf.model_type != "deepseek_mtp" or hf.architectures != ["DeepseekV32MTPModel"]
            or hf.num_hidden_layers != model.num_hidden_layers
            or hf.num_nextn_predict_layers != model.num_mtp_layers
            or hf.hidden_size != model.hidden_size or hf.kv_lora_rank != model.kv_lora_rank
            or hf.qk_rope_head_dim != model.qk_rope_head_dim or hf.index_head_dim != model.index_head_dim
            or spec.draft_parallel_config.tensor_parallel_size != vllm_config.parallel_config.tensor_parallel_size):
        raise ValueError("0.30 GLM MTP native draft metadata disagrees with appended checkpoint geometry")


def vllm030_glm_cache_specs(vllm_config, model: ModelConfig):
    """Project BF16/CP1 GLM target and explicitly configured appended MTP.

    Does not construct model layers, select a provider or establish capacity.
    """
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole

    if not isinstance(vllm_config, VllmConfig) or not isinstance(model, ModelConfig):
        raise TypeError("0.30 cache projection requires real VllmConfig and ModelConfig")
    if model.model_id not in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3"):
        raise NotImplementedError("0.30 cache projection requires an accurate GLM config")
    parallel = vllm_config.parallel_config
    if (parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or (parallel.enable_expert_parallel and not vllm030_supports_expert_parallel_cache(
                model, parallel.tensor_parallel_size))
            or vllm_config.attention_config.hisparse_config is not None):
        raise NotImplementedError("0.30 GLM cache projection requires PP1/CP1/DP1 and EP-off or EP8")
    if vllm_config.speculative_config is not None:
        validate_vllm030_glm_mtp_config(vllm_config, model)
    if (vllm_config.model_config.dtype != torch.bfloat16
            or vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16")):
        raise NotImplementedError("0.30 GLM cache projection requires BF16 latent storage")
    hf = vllm_config.model_config.hf_text_config
    for name, expected in (("hidden_size", model.hidden_size),
                           ("num_hidden_layers", model.num_hidden_layers),
                           ("kv_lora_rank", model.kv_lora_rank),
                           ("qk_rope_head_dim", model.qk_rope_head_dim),
                           ("index_head_dim", model.index_head_dim)):
        if getattr(hf, name) != expected:
            raise ValueError(f"0.30 model config disagrees with source cache field {name}")
    if tuple(hf.indexer_types) != model.indexer_types or model.index_head_dim % 128:
        raise ValueError("0.30 indexer cache geometry disagrees with source metadata")
    block_size = vllm_config.cache_config.block_size
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("0.30 cache block size must be explicit and positive")
    specs = {}
    # nvidia/mtp.py appends layer78; attention.py always constructs its
    # indexer, independent of the ordinary decoder's indexer_types pattern.
    indexer_types = model.indexer_types + (() if vllm_config.speculative_config is None else ("full",))
    for layer, indexer_type in enumerate(indexer_types):
        if indexer_type not in ("full", "shared"):
            raise ValueError("0.30 GLM cache source requires full/shared indexer types")
        prefix = f"model.layers.{layer}.self_attn"
        # MLAAttention.get_kv_cache_spec, ordinary BF16/CP1 source branch.
        specs[prefix + ".attn"] = MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1,
            head_size=model.kv_lora_rank + model.qk_rope_head_dim,
            dtype=torch.bfloat16,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
            is_index_group_leader=indexer_type == "full",
        )
        if indexer_type == "full":
            # DeepseekV32IndexerCache.get_kv_cache_spec: packed FP8 values
            # plus FP32 group scales form a uint8 row, with INDEXER role.
            specs[prefix + ".indexer.k_cache"] = MLAAttentionSpec(
                block_size=block_size, num_kv_heads=1,
                head_size=model.index_head_dim + model.index_head_dim // 128 * 4,
                dtype=torch.uint8, cache_role=SparseCacheRole.INDEXER,
            )
    return specs


def vllm030_kimi_k3_cache_specs(vllm_config, model: ModelConfig):
    """K3 text cache source projection, not instantiated backend/cache state.

    Both KDA and custom MLA register the self_attn prefix itself. Their parent
    is language_model.model.layers in the actual multimodal constructor.
    Preserve real MambaSpec state geometry/dtypes and the MLA PE subspace
    even though rotation is disabled. No page padding is invented here.
    """
    import torch
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator, MambaStateDtypeCalculator,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, MambaSpec

    if not isinstance(vllm_config, VllmConfig) or not isinstance(model, ModelConfig):
        raise TypeError("K3 cache source projection requires real VllmConfig and ModelConfig")
    if model.model_id != "moonshotai/Kimi-K3" or model.kimi_k3_semantics is None:
        raise NotImplementedError("K3 cache source projection requires exact hybrid Kimi-K3")
    native_model = vllm_config.model_config
    cache, parallel = vllm_config.cache_config, vllm_config.parallel_config
    if native_model is None or native_model.dtype != torch.bfloat16 or not native_model.is_hybrid:
        raise ValueError("K3 native config must be BF16 and hybrid")
    if (parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1
            or parallel.enable_expert_parallel or parallel.use_sequence_parallel_moe
            or vllm_config.speculative_config is not None or cache.use_kda_recoverssm):
        raise NotImplementedError("K3 cache projection requires PP/CP/DP/SP1, EP-off and MTP-off")
    if cache.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("K3 source MLA cache projection requires BF16")
    if cache.enable_prefix_caching or cache.mamba_cache_mode != "none":
        raise NotImplementedError("K3 source state placement currently requires prefix caching off and mamba mode none")
    for name in ("block_size", "mamba_block_size"):
        if type(getattr(cache, name)) is not int or getattr(cache, name) <= 0:
            raise ValueError("K3 source cache requires explicit positive attention and mamba block sizes")
    hf, source = native_model.hf_text_config, model.kimi_k3_semantics
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"):
        if getattr(hf, name) != getattr(model, name):
            raise ValueError(f"K3 native model disagrees with source field {name}")
    for name, expected in (("num_heads", source.kda_num_heads), ("head_dim", source.kda_head_dim),
                           ("short_conv_kernel_size", source.short_conv_kernel_size),
                           ("use_full_rank_gate", source.use_full_rank_gate)):
        if hf.linear_attn_config[name] != expected:
            raise ValueError(f"K3 native model disagrees with KDA source field {name}")
    native_types = tuple("kda" if i + 1 in hf.linear_attn_config["kda_layers"] else "mla_nope"
                         for i in range(model.num_hidden_layers))
    if native_types != source.layer_attention_types:
        raise ValueError("K3 native model disagrees with source hybrid layer partition")
    tp = parallel.tensor_parallel_size
    if source.kda_num_heads % tp or model.num_attention_heads % tp:
        raise ValueError("K3 source attention heads must divide over TP")
    shapes = MambaStateShapeCalculator.kda_state_shape(
        tp, source.kda_num_heads, source.kda_head_dim,
        conv_kernel_size=source.short_conv_kernel_size, num_spec=0)
    dtypes = MambaStateDtypeCalculator.kda_state_dtype(
        native_model.dtype, cache.mamba_cache_dtype, cache.mamba_ssm_cache_dtype)
    specs = {}
    for layer, kind in enumerate(source.layer_attention_types):
        name = f"language_model.model.layers.{layer}.self_attn"
        if kind == "kda":
            specs[name] = MambaSpec(
                block_size=cache.mamba_block_size, shapes=shapes, dtypes=dtypes,
                page_size_padded=cache.mamba_page_size_padded,
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                mamba_cache_mode=cache.mamba_cache_mode, num_speculative_blocks=0,
                tp_replicated=False)
        elif kind == "mla_nope":
            specs[name] = MLAAttentionSpec(
                block_size=cache.block_size, num_kv_heads=1,
                head_size=model.kv_lora_rank + model.qk_rope_head_dim,
                dtype=torch.bfloat16, cache_dtype_str=cache.cache_dtype,
                non_causal_multi_token_decode=False)
        else:
            raise ValueError(f"unhandled K3 source layer kind: {kind}")
    return specs


def allocate_vllm030_kimi_k3_cache(
    vllm_config, model: ModelConfig, parallelism: ParallelConfig,
    available_memory_by_rank: tuple[int, ...], *, source_root: Path,
    supported_layouts_by_rank: tuple[tuple[str, ...], ...],
):
    """Run native hybrid allocation for explicitly declared K3 source inputs.

    CPU API component only. No debug block count, actual H200 budget/provider,
    model loading or normalized Serving integration is established here.
    Upstream owns grouping, page padding, shared-pool strides and reconciliation.
    """
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("K3 allocation requires ModelConfig and ParallelConfig")
    if model.model_id != "moonshotai/Kimi-K3" or model.kimi_k3_semantics is None:
        raise NotImplementedError("K3 allocation requires exact hybrid Kimi-K3")
    if (not parallelism.is_rank_local or parallelism.pipeline_parallel_size != 1
            or parallelism.enable_expert_parallel):
        raise NotImplementedError("K3 source allocation requires PP/CP/DP/SP1 and EP-off")
    if (not isinstance(available_memory_by_rank, tuple)
            or len(available_memory_by_rank) != parallelism.worker_count
            or any(type(value) is not int or value <= 0 for value in available_memory_by_rank)):
        raise ValueError("K3 allocator requires one explicit positive byte budget per rank")
    if (not isinstance(supported_layouts_by_rank, tuple)
            or len(supported_layouts_by_rank) != parallelism.worker_count
            or any(not isinstance(layouts, tuple) or not layouts
                   or any(not isinstance(name, str) or not name for name in layouts)
                   or len(set(layouts)) != len(layouts) for layouts in supported_layouts_by_rank)
            or any(layouts != supported_layouts_by_rank[0] for layouts in supported_layouts_by_rank)):
        raise ValueError("K3 allocator requires explicit agreeing layout capabilities on every rank")
    from src.serving.backend.vllm_scheduler_process import vllm030_source_identity
    vllm030_source_identity(source_root)
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs, generate_scheduler_kv_cache_config
    from vllm.platforms import current_platform

    if (not isinstance(vllm_config, VllmConfig) or vllm_config.device_config.device_type != "cpu"
            or not current_platform.is_cpu()):
        raise NotImplementedError("K3 allocation requires the real CPU VllmConfig")
    if vllm_config.parallel_config.tensor_parallel_size != parallelism.tensor_parallel_size:
        raise ValueError("K3 native TP disagrees with declared worker ranks")
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        raise ValueError("K3 source allocation rejects debug block overrides")
    if vllm_config.attention_config.hisparse_config is not None:
        raise NotImplementedError("K3 source allocation does not support HiSparse")
    if any(name not in KVCacheLayout.__members__ for name in supported_layouts_by_rank[0]):
        raise ValueError("K3 allocation requires canonical native layout names")
    specs = vllm030_kimi_k3_cache_specs(vllm_config, model)
    resolve_kv_cache_layout(vllm_config, [list(x) for x in supported_layouts_by_rank], specs.values())
    if vllm_config.cache_config.get_resolved_kv_cache_layout().name not in supported_layouts_by_rank[0]:
        raise ValueError("K3 resolved layout conflicts with declared worker capabilities")
    workers = get_kv_cache_configs(vllm_config, [dict(specs) for _ in available_memory_by_rank],
                                  list(available_memory_by_rank))
    if len(workers) != parallelism.worker_count:
        raise RuntimeError("K3 allocator returned inconsistent worker count")
    for worker, budget in zip(workers, available_memory_by_rank, strict=True):
        owners = [name for group in worker.kv_cache_groups for name in group.layer_names]
        if len(owners) != len(specs) or set(owners) != set(specs):
            raise RuntimeError("K3 hybrid grouping lost or duplicated cache owners")
        # Generic non-HiSparse groups overlay one shared backing pool. These
        # descriptors share size, rather than representing independent allocations.
        sizes = {tensor.size for tensor in worker.kv_cache_tensors}
        if len(sizes) != 1 or not 0 < next(iter(sizes)) <= budget:
            raise RuntimeError("K3 allocator violated the shared-pool byte budget")
    scheduler_cache = generate_scheduler_kv_cache_config(workers)
    if any(worker.num_blocks != scheduler_cache.num_blocks for worker in workers):
        raise RuntimeError("K3 allocator failed worker block-count reconciliation")
    return workers, scheduler_cache


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


def allocate_vllm030_kv_cache(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    kv_block_size_tokens: int,
    max_model_len: int,
    available_memory_by_rank: tuple[int, ...],
    scheduler_config,
    python_executable: Path,
    source_root: Path,
    max_batched_tokens: int,
    max_requests: int,
    supported_layouts_by_rank: tuple[tuple[str, ...], ...],
) -> VLLMKVAllocation:
    """Native 0.30 callback for the analytical planner's allocator protocol.

    Bind the SDK/limits/layout keywords with functools.partial. The existing
    guarded CPU process owns all native objects and is closed before returning.
    Its caller-supplied budgets are not proof of normal available-KV. CPU
    configuration rewrites must not silently change analytical plan inputs.
    """
    from src.serving.backend.vllm import VLLMSchedulerConfig
    from src.serving.backend.vllm_scheduler_process import VLLM030SchedulerProcess
    from src.serving.workload.model_config import load_model_config

    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("0.30 allocator callback requires ModelConfig and ParallelConfig")
    if not isinstance(scheduler_config, VLLMSchedulerConfig) or scheduler_config.scheduler_version != "0.30.0":
        raise ValueError("0.30 allocator callback requires an explicit 0.30 Scheduler config")
    if (not parallelism.is_rank_local or (parallelism.pipeline_parallel_size != 1 and not (
            not parallelism.enable_expert_parallel
            and is_validated_qwen3_pipeline_model_scope(model, parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size)))
            or (parallelism.enable_expert_parallel and not vllm030_supports_expert_parallel_cache(
                model, parallelism.tensor_parallel_size))):
        raise NotImplementedError("0.30 allocator callback requires PP1 or uniform Qwen3 TP1 PP/TP2 PP2, and supported expert placement")
    configured_model = load_model_config(scheduler_config.model)
    if (configured_model != model
            or scheduler_config.tensor_parallel_size != parallelism.tensor_parallel_size
            or scheduler_config.pipeline_parallel_size != parallelism.pipeline_parallel_size
            or scheduler_config.enable_expert_parallel != parallelism.enable_expert_parallel
            or scheduler_config.max_model_len != max_model_len):
        raise ValueError("0.30 allocator callback model/parallelism/length disagrees with analytical inputs")
    process = VLLM030SchedulerProcess(
        scheduler_config, python_executable=python_executable, source_root=source_root,
        pool_name="analytical-kv-allocation", max_batched_tokens=max_batched_tokens,
        max_requests=max_requests, kv_block_size_tokens=kv_block_size_tokens,
        available_memory_by_rank=available_memory_by_rank,
        supported_layouts_by_rank=supported_layouts_by_rank)
    try:
        allocation = VLLMKVAllocation(**process.runtime_info["native_allocation"])
        effective = process.backend_config
        if (effective.max_batched_tokens != max_batched_tokens
                or effective.max_requests != max_requests
                or allocation.kv_block_size_tokens != kv_block_size_tokens
                or allocation.max_model_len != max_model_len):
            raise ValueError(
                "0.30 effective SDK configuration changed analytical allocator inputs: "
                f"batch={effective.max_batched_tokens}, requests={effective.max_requests}, "
                f"block={allocation.kv_block_size_tokens}, model_len={allocation.max_model_len}")
        return allocation
    finally:
        process.close()


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
