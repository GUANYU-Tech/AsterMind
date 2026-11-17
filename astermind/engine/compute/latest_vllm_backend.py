"""vLLM 0.31 CPU Scheduler bridge for the existing BF16 GQA Cells."""
from dataclasses import dataclass
import importlib.metadata
import multiprocessing
import os
from pathlib import Path
import traceback
from types import SimpleNamespace

from src.serving.kv_cache.kv_cache_events import KVCacheEvent
from src.serving.workload.model_config import load_model_config


@dataclass(frozen=True)
class LatestRuntimeInfo:
    version: str
    source: str
    package_python_file_count: int
    package_python_manifest_sha256: str
    platform: str = "cpu"
    adapter: str = "external_vllm_0_31_sync_v2"
    upstream_exact_environment: bool = False
    capacity: dict | None = None


def construct_scheduler(config, requested, storage_options=None):
    import torch
    from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig, StructuredOutputsConfig
    from vllm.config.ec_manager_config import EncoderCacheManagerConfig
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
    from vllm.v1.structured_output import StructuredOutputManager

    model = load_model_config(requested.model)
    tp, pp = requested.tensor_parallel_size, requested.pipeline_parallel_size
    if (config.role != "colocated" or config.scheduler_pipeline_depth != 1
            or ((tp, pp) not in ((1, 1), (2, 1), (2, 2))
                and not (tp == 1 and pp >= 2 and model.model_type == "qwen3"
                         and not model.tie_word_embeddings))
            or requested.async_scheduling
            or requested.mtp_num_speculative_tokens
            or model.attention_layout != "standard_gqa" or model.compute_dtype != "bf16"):
        raise NotImplementedError("latest adapter supports dense BF16 colocated sync TP1/PP1, TP2/PP1 and TP2/PP2 only")
    if tp > 1 and (model.model_type not in ("qwen3", "qwen3_moe") or model.weight_dtype != "bf16"):
        raise NotImplementedError("multi-GPU bridge requires BF16 Qwen3/Qwen3MoE")
    if model.num_experts:
        from src.serving.workload.model_config import is_validated_qwen3_moe_ep2_model_scope
        if (not requested.enable_expert_parallel or (tp, pp) != (2, 1)
                or not is_validated_qwen3_moe_ep2_model_scope(model)):
            raise NotImplementedError("MoE bridge supports only native Qwen3MoE TP2/EP2")
    elif requested.enable_expert_parallel:
        raise ValueError("expert parallelism requires a MoE model")
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
    layer_ranges = pipeline_layer_ranges_for_model(model, pp)
    if len({end - start for start, end in layer_ranges}) != 1:
        raise NotImplementedError("multi-GPU bridge requires uniform stage KV geometry")
    layer_start, layer_end = layer_ranges[0]
    scheduler_config = SchedulerConfig(
        max_model_len=requested.max_model_len, is_encoder_decoder=False,
        max_num_batched_tokens=config.max_batched_tokens, max_num_seqs=config.max_requests,
        enable_chunked_prefill=True, async_scheduling=False, policy="fcfs",
        long_prefill_token_threshold=0, watermark=(storage_options or {}).get("watermark", 0.0),
        # Keep the new version's native default; record this difference.
        scheduler_reserve_full_isl=True,
    )
    cache = CacheConfig(block_size=config.kv_block_size_tokens,
                        enable_prefix_caching=requested.enable_prefix_caching)
    # Explicit layer/token/head layout for the CPU allocator; no GPU backend
    # is loaded to resolve it. This does not inject addresses into M-L.
    cache.kv_cache_layout = "LBNHC"
    cache.num_gpu_blocks = config.num_kv_blocks
    cache.num_gpu_blocks_override = config.num_kv_blocks
    view = SimpleNamespace(
        max_model_len=requested.max_model_len, original_max_model_len=requested.max_model_len,
        vocab_size=model.vocab_size, dtype=torch.bfloat16,
        is_encoder_decoder=False, is_diffusion=False, uses_mrope=False,
        supports_multimodal_inputs=False, skip_tokenizer_init=True, return_sampling_mask=False,
    )
    parallel = ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp,
                              enable_expert_parallel=requested.enable_expert_parallel)
    bundle = SimpleNamespace(
        model_config=view, cache_config=cache, scheduler_config=scheduler_config,
        parallel_config=parallel,
        structured_outputs_config=StructuredOutputsConfig(), ec_manager_config=EncoderCacheManagerConfig(),
        attention_config=SimpleNamespace(hisparse_config=None),
        observability_config=SimpleNamespace(per_request_spec_decode_metrics=None,
                                             kv_cache_metrics=False, enable_mfu_metrics=False),
        lora_config=None, kv_events_config=None, kv_transfer_config=None, ec_transfer_config=None,
        speculative_config=None, is_mm_encoder_only=False, num_speculative_tokens=0,
        num_lookahead_tokens=0, num_prefill_lookahead_tokens=0, max_in_flight_tokens=0,
        max_concurrent_batches=1, use_v2_model_runner=False,
        aux_output_config=SimpleNamespace(enabled=False),
    )
    spec = FullAttentionSpec(block_size=config.kv_block_size_tokens,
                             num_kv_heads=model.kv_heads_per_tensor_parallel_rank(tp),
                             head_size=model.head_dim, dtype=torch.bfloat16)
    names = [f"model.layers.{i}.self_attn.attn" for i in range(layer_start, layer_end)]
    kv = KVCacheConfig(config.num_kv_blocks,
                       [KVCacheTensor(size=config.num_kv_blocks * spec.page_size_bytes, layers=[name],
                                      layer_stride=config.num_kv_blocks * spec.page_size_bytes,
                                      block_stride=spec.page_size_bytes) for name in names],
                       [KVCacheGroupSpec(names, spec)])
    capacity = dict((storage_options or {}).get("capacity", {}))
    if capacity.get("source_mode") == "analytical_current_vllm_allocator":
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        cache.num_gpu_blocks_override = None
        kv = get_kv_cache_configs(bundle, [{name: spec for name in names}],
                                  [capacity["available_kv_bytes"]])[0]
        cache.num_gpu_blocks = kv.num_blocks
    capacity.update(num_kv_blocks=kv.num_blocks,
                    logical_kv_block_tokens=config.kv_block_size_tokens,
                    tensor_parallel_size=tp, pipeline_parallel_size=pp,
                    data_parallel_size=1, data_parallel_rank=0,
                    kv_geometry_layer_range=[layer_start, layer_end],
                    kv_cache_layout=cache.kv_cache_layout,
                    allocated_kv_tensor_bytes=sum(t.size for t in kv.kv_cache_tensors),
                    native_allocator="vllm.v1.core.kv_cache_utils.get_kv_cache_configs"
                    if capacity.get("source_mode") == "analytical_current_vllm_allocator" else "explicit_debug_geometry")
    if sum(t.size for t in kv.kv_cache_tensors) != (kv.num_blocks
            * config.kv_block_size_tokens * config.kv_bytes_per_token):
        raise ValueError("latest Scheduler KV geometry differs from native M-L model")
    scheduler = Scheduler(bundle, kv, StructuredOutputManager(bundle), config.kv_block_size_tokens)
    scheduler.orbitserve_capacity = capacity
    return scheduler


def normalize_sampled_feedback(step, sampled_token_ids, vocab_size):
    """Validate explicit worker feedback; retain original dummy replay by default."""
    if sampled_token_ids is None:
        return {rid:[0]*count for rid,count in step.generated_tokens.items()}
    if type(vocab_size) is not int or vocab_size<1:
        raise ValueError("positive feedback vocabulary required")
    if type(sampled_token_ids) is not dict or set(sampled_token_ids)!=set(step.generated_tokens):
        raise ValueError("sampled feedback must identify every scheduled request exactly once")
    for rid,values in sampled_token_ids.items():
        if type(values) is not list or len(values)!=step.generated_tokens[rid]:
            raise ValueError("sampled feedback count disagrees with normalized Scheduler step")
        if any(type(token) is not int or not 0<=token<vocab_size for token in values):
            raise ValueError("sampled feedback token outside model vocabulary")
    return {rid:list(sampled_token_ids[rid]) for rid in step.generated_tokens}


class LatestControlPlane:
    def has_work(self):
        return bool(self.scheduler.requests)

    def add_requests(self, requests):
        for request in requests:
            self.add_request(request)

    def blocks(self, request_id):
        groups = self.scheduler.kv_cache_manager.get_block_ids(request_id)
        if len(groups) != 1:
            raise RuntimeError("adapter requires exactly one KV group")
        return tuple(groups[0])

    def kv_event(self, request_id, action, start_block, end_block):
        start = start_block * self.config.kv_block_size_tokens
        end = end_block * self.config.kv_block_size_tokens
        return KVCacheEvent(request_id, action, "hbm", start, end,
                            (end-start) * self.config.kv_bytes_per_token)

    def complete_step(self, step, now, *, sampled_token_ids=None, **flags):
        from vllm.v1.outputs import ModelRunnerOutput
        if any(flags.values()) or self.pending is None or self.pending[0] != step:
            raise RuntimeError("invalid feedback or real model execution requested")
        ids = list(step.generated_tokens)
        feedback=normalize_sampled_feedback(step,sampled_token_ids,self.scheduler.vllm_config.model_config.vocab_size)
        output = ModelRunnerOutput(req_ids=ids, req_id_to_index={rid:i for i,rid in enumerate(ids)},
                                   sampled_token_ids=[feedback[rid] for rid in ids])
        result = self.scheduler.update_from_output(self.pending[1], output)
        emitted = {rid: 0 for rid in ids}
        actual_finished = set()
        for batch in result.values():
            for row in batch.outputs:
                emitted[row.request_id] += len(row.new_token_ids)
                if row.finish_reason is not None:
                    actual_finished.add(row.request_id)
        if emitted != dict(step.generated_tokens) or actual_finished != set(step.finished_request_ids):
            raise RuntimeError("native Scheduler feedback disagrees with normalized M-L step")
        for rid in actual_finished:
            self.validate_finished_storage(rid)
        self.pending = None
        return 0.0

    def validate_finished_storage(self, rid):
        if rid in self.scheduler.requests or self.blocks(rid):
            raise RuntimeError("finished request was not released")


def worker(pipe, config, requested, storage_options):
    try:
        os.environ.update(MIDLAYER_VLLM_SCHEDULER_WORKER="1", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
        from src.serving.backend.vllm_scheduler_process import (
            _install_cpu_device_guard, _package_python_manifest, _verify_no_accelerator_access,
        )
        baseline_libraries = _install_cpu_device_guard()
        import importlib.util
        import json
        package = Path(importlib.util.find_spec("vllm").origin).parent
        count, manifest = _package_python_manifest(package)
        identity = json.loads((Path(__file__).parent / "scheduler_runtime/vllm_0_31_identity.json").read_text())
        if importlib.metadata.version("vllm") != "0.31.0" or (count, manifest) != (
                identity["package_python_file_count"], identity["package_python_manifest_sha256"]):
            raise RuntimeError("installed latest vLLM differs from the verified official wheel")
        from .native_cell_storage import StorageControlPlane
        plane = StorageControlPlane(config, requested, storage_options)
        _verify_no_accelerator_access(baseline_libraries)
        pipe.send(("ok", LatestRuntimeInfo("0.31.0", identity["url"], count, manifest,
                                           capacity=plane.scheduler.orbitserve_capacity)))
        while True:
            name, args, kwargs = pipe.recv()
            if name == "close":
                pipe.send(("ok", None))
                break
            result = getattr(plane, name)(*args, **kwargs)
            if name == "complete_step":
                _verify_no_accelerator_access(baseline_libraries)
            pipe.send(("ok", result))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()


class LatestSchedulerProcess:
    def __init__(self, config, requested, *, storage_options):
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child, config, requested, storage_options), daemon=True)
        self.closed = False
        self.process.start()
        child.close()
        try:
            # CPU-only vLLM imports and integrity validation can exceed an RPC
            # deadline on a cold WSL filesystem. Preserve the dynamic-link
            # platform's separate startup budget; ordinary RPCs retain 60 s.
            self.runtime_info = self.receive(timeout=180)
        except BaseException:
            self.close()
            raise

    def receive(self, timeout=60):
        if not self.pipe.poll(timeout):
            raise TimeoutError(f"latest Scheduler worker exceeded {timeout} seconds")
        status, value = self.pipe.recv()
        if status != "ok":
            raise RuntimeError(value)
        return value

    def rpc(self, name, *args, **kwargs):
        if self.closed:
            raise RuntimeError("closed latest Scheduler")
        self.pipe.send((name, args, kwargs))
        return self.receive()

    def add_request(self, request):
        return self.rpc("add_request", request)

    def add_requests(self, requests):
        return self.rpc("add_requests", requests)

    def plan_step(self, now):
        return self.rpc("plan_step", now)

    def complete_step(self, step, now, **flags):
        return self.rpc("complete_step", step, now, **flags)

    def has_work(self):
        return self.rpc("has_work")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            try:
                self.pipe.send(("close", (), {}))
                if self.pipe.poll(5):
                    self.pipe.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        self.pipe.close()
