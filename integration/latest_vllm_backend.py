"""Explicit vLLM 0.31 Scheduler bridge for synchronous BF16 GQA.

Legacy scopes are retained. Opt-in Qwen3-MoE TP/PP/EP uses a declared native
source extension; per-DP Scheduler identities feed the external coordinator.
The external DP coordinator composes GPU work. Optional storage control plane
enables declared same-session prefix caching and simulated PD connector events.
No GPU NIXL workers, MTP or offload execution is provided by this bridge.
Only real vLLM makes admission, scheduling, preemption and KV decisions.
Default dummy token feedback models a fixed output length, as in upstream M-L.
Optional explicit sampled-token feedback is validated before native update.
"""

import multiprocessing
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduledRequest,
    ScheduleStep,
)
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
    from vllm.config import (
        CacheConfig,
        ParallelConfig,
        SchedulerConfig,
        StructuredOutputsConfig,
    )
    from vllm.config.ec_manager_config import EncoderCacheManagerConfig
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    model = load_model_config(requested.model)
    tp, pp = requested.tensor_parallel_size, requested.pipeline_parallel_size
    dcp_size, dcp_interleave = 1, 1
    if (
        storage_options is not None
        and "decode_context_parallel_size" in storage_options
    ):
        dcp_size = storage_options["decode_context_parallel_size"]
        dcp_interleave = storage_options["cp_kv_cache_interleave_size"]
        from dcp_layout import GQADCPLayout

        GQADCPLayout(
            model.num_attention_heads,
            model.num_key_value_heads,
            tp,
            dcp_size,
            config.kv_block_size_tokens,
            dcp_interleave,
        )
        if requested.enable_prefix_caching:
            raise NotImplementedError(
                "DCP prefix/session integration is not yet connected"
            )
    extended = (
        storage_options is not None
        and "qwen_parallel_extension" in storage_options
        and storage_options["qwen_parallel_extension"] is True
    )
    dp_size, dp_rank = 1, 0
    if storage_options is not None and (
        "data_parallel_size" in storage_options
        or "data_parallel_rank" in storage_options
    ):
        if not {"data_parallel_size", "data_parallel_rank"} <= storage_options.keys():
            raise ValueError("DP Scheduler identity needs both size and rank")
        dp_size, dp_rank = (
            storage_options["data_parallel_size"],
            storage_options["data_parallel_rank"],
        )
        if (
            type(dp_size) is not int
            or type(dp_rank) is not int
            or dp_size < 1
            or not 0 <= dp_rank < dp_size
        ):
            raise ValueError("invalid local DP Scheduler identity")
        if not extended or model.model_type != "qwen3_moe":
            raise NotImplementedError(
                "DP Scheduler bridge requires the explicit Qwen3-MoE extension"
            )
    if (
        config.role != "colocated"
        or config.scheduler_pipeline_depth != 1
        or (not extended and (tp, pp) not in ((1, 1), (2, 1), (2, 2))
            and not (tp == 1 and pp >= 2 and model.model_type == "qwen3"
                     and not model.tie_word_embeddings))
        or requested.async_scheduling
        or requested.mtp_num_speculative_tokens
        or model.attention_layout != "standard_gqa"
        or model.compute_dtype != "bf16"
    ):
        raise NotImplementedError(
            "latest adapter requires supported BF16 colocated synchronous TP/PP emission"
        )
    if tp > 1 and (
        model.model_type not in ("qwen3", "qwen3_moe") or model.weight_dtype != "bf16"
    ):
        raise NotImplementedError("multi-GPU bridge requires BF16 Qwen3/Qwen3MoE")
    if model.num_experts:
        from src.serving.workload.model_config import (
            is_validated_qwen3_moe_ep2_model_scope,
        )

        if not extended and (
            not requested.enable_expert_parallel
            or (tp, pp) != (2, 1)
            or not is_validated_qwen3_moe_ep2_model_scope(model)
        ):
            raise NotImplementedError(
                "MoE bridge supports only native Qwen3MoE TP2/EP2"
            )
    elif requested.enable_expert_parallel:
        raise ValueError("expert parallelism requires a MoE model")
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_model

    layer_ranges = pipeline_layer_ranges_for_model(model, pp)
    if len({end - start for start, end in layer_ranges}) != 1:
        raise NotImplementedError("multi-GPU bridge requires uniform stage KV geometry")
    layer_start, layer_end = layer_ranges[0]
    scheduler_config = SchedulerConfig(
        max_model_len=requested.max_model_len,
        is_encoder_decoder=False,
        max_num_batched_tokens=config.max_batched_tokens,
        max_num_seqs=config.max_requests,
        enable_chunked_prefill=True,
        async_scheduling=False,
        policy="fcfs",
        long_prefill_token_threshold=0,
        watermark=(storage_options or {}).get("watermark", 0.0),
        # Keep the new version's native default; record this difference.
        scheduler_reserve_full_isl=True,
    )
    cache = CacheConfig(
        block_size=config.kv_block_size_tokens,
        enable_prefix_caching=requested.enable_prefix_caching,
    )
    # Explicit layer/token/head layout for the CPU allocator; no GPU backend
    # is loaded to resolve it. This does not inject addresses into M-L.
    cache.kv_cache_layout = "LBNHC"
    cache.num_gpu_blocks = config.num_kv_blocks
    cache.num_gpu_blocks_override = config.num_kv_blocks
    view = SimpleNamespace(
        max_model_len=requested.max_model_len,
        original_max_model_len=requested.max_model_len,
        vocab_size=model.vocab_size,
        dtype=torch.bfloat16,
        is_encoder_decoder=False,
        is_diffusion=False,
        uses_mrope=False,
        supports_multimodal_inputs=False,
        skip_tokenizer_init=True,
        return_sampling_mask=False,
    )
    parallel_arguments = {
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "enable_expert_parallel": requested.enable_expert_parallel,
        "data_parallel_size": dp_size,
        "data_parallel_rank": dp_rank,
    }
    if (
        storage_options is not None
        and "decode_context_parallel_size" in storage_options
    ):
        parallel_arguments.update(
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=dcp_interleave,
        )
    parallel = ParallelConfig(**parallel_arguments)
    bundle = SimpleNamespace(
        model_config=view,
        cache_config=cache,
        scheduler_config=scheduler_config,
        parallel_config=parallel,
        structured_outputs_config=StructuredOutputsConfig(),
        ec_manager_config=EncoderCacheManagerConfig(),
        attention_config=SimpleNamespace(hisparse_config=None),
        observability_config=SimpleNamespace(
            per_request_spec_decode_metrics=None,
            kv_cache_metrics=False,
            enable_mfu_metrics=False,
        ),
        lora_config=None,
        kv_events_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
        is_mm_encoder_only=False,
        num_speculative_tokens=0,
        num_lookahead_tokens=0,
        num_prefill_lookahead_tokens=0,
        max_in_flight_tokens=0,
        max_concurrent_batches=1,
        use_v2_model_runner=False,
        aux_output_config=SimpleNamespace(enabled=False),
    )
    spec = FullAttentionSpec(
        block_size=config.kv_block_size_tokens,
        num_kv_heads=model.kv_heads_per_tensor_parallel_rank(tp),
        head_size=model.head_dim,
        dtype=torch.bfloat16,
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(layer_start, layer_end)]
    kv = KVCacheConfig(
        config.num_kv_blocks,
        [
            KVCacheTensor(
                size=config.num_kv_blocks * spec.page_size_bytes,
                layers=[name],
                layer_stride=config.num_kv_blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
            for name in names
        ],
        [KVCacheGroupSpec(names, spec)],
    )
    capacity = dict((storage_options or {}).get("capacity", {}))
    if capacity.get("source_mode") == "analytical_current_vllm_allocator":
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

        cache.num_gpu_blocks_override = None
        kv = get_kv_cache_configs(
            bundle, [{name: spec for name in names}], [capacity["available_kv_bytes"]]
        )[0]
        cache.num_gpu_blocks = kv.num_blocks
    capacity.update(
        num_kv_blocks=kv.num_blocks,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=parallel.cp_kv_cache_interleave_size,
        logical_kv_block_tokens=config.kv_block_size_tokens * dcp_size,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        kv_geometry_layer_range=[layer_start, layer_end],
        kv_cache_layout=cache.kv_cache_layout,
        allocated_kv_tensor_bytes=sum(t.size for t in kv.kv_cache_tensors),
        native_allocator="vllm.v1.core.kv_cache_utils.get_kv_cache_configs"
        if capacity.get("source_mode") == "analytical_current_vllm_allocator"
        else "explicit_debug_geometry",
    )
    if sum(t.size for t in kv.kv_cache_tensors) != (
        kv.num_blocks * config.kv_block_size_tokens * config.kv_bytes_per_token
    ):
        raise ValueError("latest Scheduler KV geometry differs from native M-L model")
    scheduler = Scheduler(
        bundle, kv, StructuredOutputManager(bundle), config.kv_block_size_tokens
    )
    scheduler.orbitserve_capacity = capacity
    return scheduler


def normalize_sampled_feedback(step, sampled_token_ids, vocab_size):
    """Validate explicit worker feedback; retain original dummy replay by default."""
    if sampled_token_ids is None:
        return {rid: [0] * count for rid, count in step.generated_tokens.items()}
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("positive feedback vocabulary required")
    if type(sampled_token_ids) is not dict or set(sampled_token_ids) != set(
        step.generated_tokens
    ):
        raise ValueError(
            "sampled feedback must identify every scheduled request exactly once"
        )
    for rid, values in sampled_token_ids.items():
        if type(values) is not list or len(values) != step.generated_tokens[rid]:
            raise ValueError(
                "sampled feedback count disagrees with normalized Scheduler step"
            )
        if any(
            type(token) is not int or not 0 <= token < vocab_size for token in values
        ):
            raise ValueError("sampled feedback token outside model vocabulary")
    return {rid: list(sampled_token_ids[rid]) for rid in step.generated_tokens}


class LatestControlPlane:
    def __init__(self, config, requested):
        self.config = config
        self.scheduler = construct_scheduler(config, requested)
        self.specs = {}
        self.highwater = {}
        self.pending = None
        self.index = 0

    def add_request(self, spec):
        from vllm import SamplingParams
        from vllm.v1.request import Request

        if (
            spec.request_id in self.specs
            or spec.prefix_hit_tokens
            or spec.output_tokens < 1
        ):
            raise ValueError(
                "unique positive-output requests without prefix reuse required"
            )
        self.specs[spec.request_id] = spec
        self.highwater[spec.request_id] = 0
        request = Request(
            spec.request_id,
            [0] * spec.input_tokens,
            SamplingParams(max_tokens=spec.output_tokens, ignore_eos=True),
            None,
            arrival_time=spec.arrival_time_s,
        )
        self.scheduler.add_request(request)

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
        return KVCacheEvent(
            request_id,
            action,
            "hbm",
            start,
            end,
            (end - start) * self.config.kv_bytes_per_token,
        )

    def plan_step(self, now):
        if self.pending is not None:
            raise RuntimeError("one in-flight batch only")
        if not self.has_work():
            return None
        before = {
            rid: (r.num_computed_tokens, r.status.name, self.blocks(rid), r.num_tokens)
            for rid, r in self.scheduler.requests.items()
        }
        native = self.scheduler.schedule()
        if not native.num_scheduled_tokens:
            raise RuntimeError("real latest Scheduler made no progress")
        events = []
        preempted = []
        for rid, (computed, status, blocks, _) in before.items():
            r = self.scheduler.requests[rid]
            if status != "PREEMPTED" and r.status.name == "PREEMPTED":
                preempted.append(rid)
                if self.blocks(rid):
                    raise RuntimeError("preempted request retains KV blocks")
                if blocks:
                    events.append(
                        self.kv_event(rid, KVCacheAction.RELEASE, 0, len(blocks))
                    )
        scheduled, generated, finished = [], {}, []
        for rid, count in native.num_scheduled_tokens.items():
            computed, status, old_blocks, generation_boundary = before[rid]
            r = self.scheduler.requests[rid]
            if r.num_computed_tokens != computed + count:
                raise RuntimeError(
                    "unexpected latest Scheduler computed-token transition"
                )
            phase = "prefill" if computed < self.specs[rid].input_tokens else "decode"
            emit = int(r.num_computed_tokens >= generation_boundary)
            recomputed = max(
                0, min(r.num_computed_tokens, self.highwater[rid]) - computed
            )
            input_count = (
                min(count, self.specs[rid].input_tokens - computed)
                if phase == "prefill"
                else 0
            )
            scheduled.append(
                ScheduledRequest(
                    rid,
                    phase,
                    input_count,
                    count - input_count,
                    computed_tokens=computed,
                    recomputed_tokens=recomputed,
                    resumed_from_preemption=status == "PREEMPTED",
                    raw_model_output_tokens=emit,
                )
            )
            generated[rid] = emit
            new_blocks = self.blocks(rid)
            if len(new_blocks) > len(old_blocks):
                events.append(
                    self.kv_event(
                        rid, KVCacheAction.ALLOCATE, len(old_blocks), len(new_blocks)
                    )
                )
            if input_count:
                events.append(
                    KVCacheEvent(
                        rid,
                        KVCacheAction.MISS,
                        "hbm",
                        computed,
                        computed + input_count,
                        input_count * self.config.kv_bytes_per_token,
                    )
                )
            self.highwater[rid] = max(self.highwater[rid], r.num_computed_tokens)
            if len(r.output_token_ids) + emit == self.specs[rid].output_tokens:
                finished.append(rid)
                if new_blocks:
                    events.append(
                        self.kv_event(rid, KVCacheAction.RELEASE, 0, len(new_blocks))
                    )
        step = BackendStep(
            ScheduleStep(
                f"{self.config.pool_name}:{self.index}",
                self.config.pool_name,
                tuple(scheduled),
                tuple(finished),
                tuple(preempted),
                "paged_varlen",
            ),
            tuple(events),
            generated,
        )
        self.index += 1
        self.pending = (step, native)
        return step

    def complete_step(self, step, now, *, sampled_token_ids=None, **flags):
        from vllm.v1.outputs import ModelRunnerOutput

        if any(flags.values()) or self.pending is None or self.pending[0] != step:
            raise RuntimeError("invalid feedback or real model execution requested")
        ids = list(step.generated_tokens)
        feedback = normalize_sampled_feedback(
            step, sampled_token_ids, self.scheduler.vllm_config.model_config.vocab_size
        )
        output = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[feedback[rid] for rid in ids],
        )
        result = self.scheduler.update_from_output(self.pending[1], output)
        emitted = {rid: 0 for rid in ids}
        actual_finished = set()
        for batch in result.values():
            for row in batch.outputs:
                emitted[row.request_id] += len(row.new_token_ids)
                if row.finish_reason is not None:
                    actual_finished.add(row.request_id)
        if emitted != dict(step.generated_tokens) or actual_finished != set(
            step.finished_request_ids
        ):
            raise RuntimeError(
                "native Scheduler feedback disagrees with normalized M-L step"
            )
        for rid in actual_finished:
            self.validate_finished_storage(rid)
        self.pending = None
        return 0.0

    def validate_finished_storage(self, rid):
        if rid in self.scheduler.requests or self.blocks(rid):
            raise RuntimeError("finished request was not released")


def worker(pipe, config, requested, storage_options=None):
    try:
        os.environ.update(
            MIDLAYER_VLLM_SCHEDULER_WORKER="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            VLLM_NO_USAGE_STATS="1",
        )
        if (
            storage_options is not None
            and "kimi_pd_control" in storage_options
            and storage_options["kimi_pd_control"]
        ):
            # Native NIXL SSM transfer requires DS. Set before importing any
            # cached layout/configuration helper; regular K3 stays unchanged.
            os.environ["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"
        from src.serving.backend.vllm_scheduler_process import (
            _install_cpu_device_guard,
            _package_python_manifest,
            _verify_no_accelerator_access,
        )

        baseline_libraries = _install_cpu_device_guard()
        import importlib.util
        import json

        package = Path(importlib.util.find_spec("vllm").origin).parent
        count, manifest = _package_python_manifest(package)
        identity = json.loads(
            (Path(__file__).parent / "runtime/vllm_0_31_identity.json").read_text()
        )
        if importlib.metadata.version("vllm") != "0.31.0" or (count, manifest) != (
            identity["package_python_file_count"],
            identity["package_python_manifest_sha256"],
        ):
            raise RuntimeError(
                "installed latest vLLM differs from the verified official wheel"
            )
        if storage_options is None:
            plane = LatestControlPlane(config, requested)
        else:
            if "target_model" in storage_options:
                if storage_options["target_model"] == "kimi_k3":
                    from kimi_scheduler import KimiControlPlane

                    if storage_options.get("kimi_pd_control"):
                        from kimi_pd import KimiPDControlPlane, KimiPDDCPControlPlane

                        plane_type = (
                            KimiPDDCPControlPlane
                            if "decode_context_parallel_size" in storage_options
                            and storage_options["decode_context_parallel_size"] > 1
                            else KimiPDControlPlane
                        )
                        plane = plane_type(config, requested, storage_options)
                    elif storage_options.get("kimi_dcp_control"):
                        from kimi_dcp import KimiDCPControlPlane

                        plane = KimiDCPControlPlane(config, requested, storage_options)
                    else:
                        plane = KimiControlPlane(config, requested, storage_options)
                elif storage_options["target_model"] == "deepseek_v4_flash":
                    if storage_options.get("v4_pd_control"):
                        from deepseek_v4_pd import V4PDControlPlane

                        plane = V4PDControlPlane(config, requested, storage_options)
                    else:
                        from deepseek_v4_scheduler import V4ControlPlane

                        plane = V4ControlPlane(config, requested, storage_options)
                else:
                    raise NotImplementedError("unknown target-model Scheduler bridge")
            elif (
                "decode_context_parallel_size" in storage_options
                and storage_options["decode_context_parallel_size"] > 1
            ):
                from dcp_scheduler import DCPControlPlane

                plane = DCPControlPlane(config, requested, storage_options)
            else:
                from native_cell_storage import StorageControlPlane

                plane = StorageControlPlane(config, requested, storage_options)
        _verify_no_accelerator_access(baseline_libraries)
        pipe.send(
            (
                "ok",
                LatestRuntimeInfo(
                    "0.31.0",
                    identity["url"],
                    count,
                    manifest,
                    capacity=plane.scheduler.orbitserve_capacity,
                ),
            )
        )
        while True:
            name, args, kwargs = pipe.recv()
            if name == "close":
                pipe.send(("ok", None))
                break
            result = getattr(plane, name)(*args, **kwargs)
            if name in (
                "complete_step",
                "configure_dp_router",
                "dp_choose_route",
                "dp_finish_routes",
            ) or name.startswith(("pd_", "dcp_", "kimi_pd_", "kimi_dcp_", "v4_pd_")):
                _verify_no_accelerator_access(baseline_libraries)
            pipe.send(("ok", result))
    except BaseException:  # noqa: BLE001 - report worker failures before closing the pipe
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()


class LatestSchedulerProcess:
    def __init__(
        self, config, requested, *, worker_target=worker, storage_options=None
    ):
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe()
        args = (
            (child, config, requested)
            if storage_options is None
            else (child, config, requested, storage_options)
        )
        self.process = context.Process(target=worker_target, args=args, daemon=True)
        self.closed = False
        self.process.start()
        child.close()
        try:
            # Cold imports and the official source-manifest verification can
            # exceed a normal RPC deadline on mounted development filesystems.
            # Keep execution RPCs bounded by the existing shorter deadline.
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
