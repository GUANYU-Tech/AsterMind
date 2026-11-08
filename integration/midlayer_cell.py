"""Persistent M-L cell adapter; native scheduling and kernel models stay upstream.

Scope: dense colocated TP1/PP1, Qwen3 TP2/PP1 and TP2/PP2,
Qwen3MoE TP2/EP2, synchronous depth-one, optional native Host launch,
exclusive GPUs,
declared linear-session prefix reuse and static migration via native blocks.
No table timing or replacement scheduler. Opt-in parallel extensions use
the explicitly recorded native snapshot changes. The factory delegates
DP execution to dp_cell; two-pool PD uses the separate pd_cell factory.
"""

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from inspect import getfile
from pathlib import Path

from src.kernel.hardware_spec import HWSpec

# Resolve the protocol type through the selected native backend. Upstream
# renamed schedule.backend to schedule.backend_protocol; the backend binds the
# canonical class in both snapshots, without a replacement protocol.
from src.serving.backend.vllm import BackendConfig, VLLMBackend, VLLMSchedulerConfig
from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    PipelineP2PConfig,
    WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.execution import ServingExecutionContext
from src.serving.kv_cache.kv_cache_events import KVCacheAction
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.request_gen.request_stream import RequestSpec
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs

from orbitserve.runtime.engine.cell_events import (
    DISPATCH,
    NOTIFY,
    STEP_COMPLETE,
    CellCompletion,
    CellEventLoop,
    CellRequest,
    CellTokenOutput,
)


def native_host_launch_profile_file(root: Path, hardware_id: str) -> str:
    """Resolve the two known upstream locations; never invent Host timing."""
    candidates = (
        f"configs/host_timing/host_launch_profiles/{hardware_id}.json",
        f"configs/evaluation/experiments/host_launch_profiles/{hardware_id}.json",
    )
    present = [name for name in candidates if (root / name).is_file()]
    if len(present) != 1:
        raise ValueError(
            f"expected one native Host profile for {hardware_id}, found {present}"
        )
    return present[0]


@dataclass(frozen=True)
class MidlayerCellConfig:
    root: Path
    model_file: str = "configs/model/qwen3-4b.json"
    hardware_file: str = "configs/hardware/nvidia_a800.yaml"
    hardware_id: str = "nvidia_a800"
    max_model_len: int = 128
    max_batched_tokens: int = 32
    max_requests: int = 4
    num_kv_blocks: int = 64
    kv_block_size_tokens: int = 16
    scheduler_adapter: str = "upstream_exact"
    enable_prefix_caching: bool = False
    watermark: float = 0.0
    memory_policy_file: str | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    expert_routing_seed: int | None = None
    host_launch_profile_file: str | None = None
    qwen_parallel_extension: bool = False
    data_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    cp_kv_cache_interleave_size: int = 1

    def __post_init__(self):
        for name in ("decode_context_parallel_size", "cp_kv_cache_interleave_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.tensor_parallel_size % self.decode_context_parallel_size:
            raise ValueError("DCP must divide the TP group")
        if self.decode_context_parallel_size > 1 and (
            not self.qwen_parallel_extension or self.scheduler_adapter != "vllm_0_31"
        ):
            raise NotImplementedError(
                "DCP Cell requires the explicit Qwen/vLLM 0.31 bridge"
            )
        ParallelConfig(
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            enable_expert_parallel=self.enable_expert_parallel,
            data_parallel_size=self.data_parallel_size,
        )
        if self.data_parallel_size > 1 and not self.qwen_parallel_extension:
            raise NotImplementedError(
                "DP requires the explicit Qwen parallel extension"
            )
        if not isinstance(self.qwen_parallel_extension, bool):
            raise TypeError("qwen_parallel_extension must be boolean")
        if (
            not self.qwen_parallel_extension
            and self.enable_expert_parallel
            and (self.tensor_parallel_size, self.pipeline_parallel_size) != (2, 1)
        ):
            raise NotImplementedError(
                "Cell EP currently reuses only the native TP2/EP2 Qwen3MoE slice"
            )
        if self.expert_routing_seed is not None and (
            isinstance(self.expert_routing_seed, bool)
            or not isinstance(self.expert_routing_seed, int)
            or self.expert_routing_seed < 0
        ):
            raise ValueError("expert_routing_seed must be a nonnegative integer")
        if not self.qwen_parallel_extension and (
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
        ) not in ((1, 1), (2, 1), (2, 2)):
            raise NotImplementedError("Cell supports TP1/PP1, TP2/PP1 and TP2/PP2 only")
        if (self.tensor_parallel_size, self.pipeline_parallel_size) != (
            1,
            1,
        ) and self.scheduler_adapter != "vllm_0_31":
            raise NotImplementedError(
                "multi-GPU Cell requires the explicit vLLM 0.31 bridge"
            )
        if self.qwen_parallel_extension and self.scheduler_adapter != "vllm_0_31":
            raise NotImplementedError(
                "Qwen parallel extension requires the vLLM 0.31 bridge"
            )
        if (
            isinstance(self.watermark, bool)
            or not math.isfinite(self.watermark)
            or not 0 <= self.watermark < 1
        ):
            raise ValueError("watermark must be in [0, 1)")
        if (
            self.enable_prefix_caching or self.watermark or self.memory_policy_file
        ) and self.scheduler_adapter != "vllm_0_31":
            raise NotImplementedError(
                "new storage features require the explicit vLLM 0.31 adapter"
            )


def create_midlayer_runtime(config: MidlayerCellConfig, *, with_scheduler: bool = True):
    if type(with_scheduler) is not bool:
        raise ValueError("with_scheduler must be boolean")
    if with_scheduler and config.decode_context_parallel_size > 1:
        raise NotImplementedError(
            "create DCP through MidlayerCell.create or create_dcp_scheduler"
        )
    if config.data_parallel_size > 1:
        raise NotImplementedError(
            "DP joint execution is not yet connected; use the DP Scheduler control API"
        )
    loaded_root = Path(getfile(HWSpec)).resolve().parents[2]
    if config.root.resolve() != loaded_root:
        raise ValueError(
            "Cell root differs from loaded M-L source; select ORBITServe_MIDLAYER_ROOT before imports"
        )
    model = load_model_config(config.root / config.model_file)
    if model.num_experts:
        from src.serving.workload.model_config import (
            is_validated_qwen3_moe_ep2_model_scope,
        )

        if (
            config.expert_routing_seed is None
            or config.scheduler_adapter != "vllm_0_31"
            or (
                not config.qwen_parallel_extension
                and (
                    not config.enable_expert_parallel
                    or not is_validated_qwen3_moe_ep2_model_scope(model)
                )
            )
        ):
            raise NotImplementedError(
                "MoE requires native Qwen3MoE TP2/EP2 and an explicit routing seed"
            )
    elif config.enable_expert_parallel or config.expert_routing_seed is not None:
        raise ValueError("dense Cells cannot carry expert parallelism/routing inputs")
    hardware = HWSpec(config.root / config.hardware_file)
    parallelism = ParallelConfig(
        tensor_parallel_size=config.tensor_parallel_size,
        pipeline_parallel_size=config.pipeline_parallel_size,
        enable_expert_parallel=config.enable_expert_parallel,
    )
    multi_gpu = parallelism.worker_count > 1
    if multi_gpu and (
        (config.hardware_id != "nvidia_a800" and not config.qwen_parallel_extension)
        or model.model_type not in ("qwen3", "qwen3_moe")
        or model.attention_layout != "standard_gqa"
        or model.compute_dtype != "bf16"
        or model.weight_dtype != "bf16"
    ):
        raise NotImplementedError(
            "multi-GPU Cell currently reuses BF16 Qwen3/Qwen3MoE on A800"
        )
    collectives = (
        tuple(
            CollectiveConfig(
                op,
                "ring",
                "nvlink",
                "ring",
                "main-nvlink-fabric",
                "analytical_roofline_v1",
            )
            for op in ("all_reduce", "all_gather")
        )
        if multi_gpu
        else ()
    )
    pipeline_p2p = tuple(
        PipelineP2PConfig(
            stage,
            stage + 1,
            "send_recv",
            "nvlink",
            "point_to_point",
            "main-pipeline-fabric",
            "analytical_roofline_v1",
        )
        for stage in range(config.pipeline_parallel_size - 1)
    )
    deployment = build_deployment_plan(
        DeploymentConfig(
            "colocated",
            (
                WorkerPoolConfig(
                    "main",
                    "colocated",
                    config.hardware_id,
                    parallelism,
                    ranks=tuple(range(parallelism.worker_count)),
                    collectives=collectives,
                    pipeline_p2p=pipeline_p2p,
                ),
            ),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    if config.qwen_parallel_extension:
        from src.serving.workload.build_workload import (
            validate_extended_qwen_parallelism,
        )

        validate_extended_qwen_parallelism(deployment, model)
    elif multi_gpu:
        from src.serving.simulation import validate_workload_emission_parallelism

        validate_workload_emission_parallelism(
            deployment,
            model,
            allow_single_pool_tp2=True,
            allow_single_pool_ep2=config.enable_expert_parallel,
            allow_single_pool_tp1_pp=False,
            allow_single_pool_tp2_pp2=True,
        )
    layer_start, layer_end = pipeline_layer_ranges_for_model(
        model, config.pipeline_parallel_size
    )[0]
    kv_bytes = model.kv_bytes_per_token_for_layer_range(
        config.tensor_parallel_size, layer_start, layer_end
    )
    backend_config = BackendConfig(
        "main",
        "colocated",
        config.max_batched_tokens,
        config.max_requests,
        config.num_kv_blocks,
        config.kv_block_size_tokens,
        kv_bytes,
        scheduler_pipeline_depth=1,
    )
    scheduler_config = VLLMSchedulerConfig(
        str(config.root / config.model_file),
        dtype="bfloat16",
        max_model_len=config.max_model_len,
        enable_chunked_prefill=True,
        enable_prefix_caching=config.enable_prefix_caching,
        async_scheduling=False,
        tensor_parallel_size=config.tensor_parallel_size,
        pipeline_parallel_size=config.pipeline_parallel_size,
        enable_expert_parallel=config.enable_expert_parallel,
    )
    if config.qwen_parallel_extension:
        from qwen_parallel_execution import QwenParallelExecutionContext

        execution_class = QwenParallelExecutionContext
    elif (
        config.scheduler_adapter == "vllm_0_31" and config.hardware_id == "nvidia_a800"
    ):
        from cell_execution import CellExecutionContext

        execution_class = CellExecutionContext
    else:
        execution_class = ServingExecutionContext
    host_profiles = None
    if config.host_launch_profile_file is not None:
        from src.serving.host_timing import load_host_launch_profile

        profile = load_host_launch_profile(
            config.root / config.host_launch_profile_file
        )
        if profile.hardware_id != config.hardware_id:
            raise ValueError("Host launch profile hardware differs from the Cell")
        host_profiles = {config.hardware_id: profile}
    routing = (
        ExpertRoutingInput.seeded_assumption(
            "orbitserve-explicit-routing-assumption", config.expert_routing_seed
        )
        if model.num_experts
        else ExpertRoutingInput.not_applicable()
    )
    execution = execution_class(
        model,
        deployment,
        {config.hardware_id: hardware},
        "analytical_launch",
        "midlayer",
        RuntimeInputs(routing, "paged_varlen", "uniform"),
        host_launch_profiles=host_profiles,
    )
    if not with_scheduler:
        return None, execution
    if config.scheduler_adapter == "upstream_exact":
        backend = VLLMBackend.from_model(backend_config, scheduler_config)
    elif config.scheduler_adapter == "vllm_0_31":
        from cell_storage import capacity_inputs
        from latest_vllm_backend import LatestSchedulerProcess

        process = LatestSchedulerProcess(
            backend_config,
            scheduler_config,
            storage_options={
                "watermark": config.watermark,
                "capacity": capacity_inputs(config, model, hardware),
                "qwen_parallel_extension": config.qwen_parallel_extension,
            },
        )
        backend_config = replace(
            backend_config, num_kv_blocks=process.runtime_info.capacity["num_kv_blocks"]
        )
        backend = VLLMBackend(backend_config, process)
    elif config.scheduler_adapter == "official_0_11":
        from official_vllm_backend import official_process

        backend = VLLMBackend(
            backend_config, official_process(backend_config, scheduler_config)
        )
    else:
        raise ValueError("unknown Scheduler adapter")
    return backend, execution


class MidlayerCell:
    def __init__(
        self,
        cell_id: str,
        loop: CellEventLoop,
        backend,
        execution,
        on_complete: Callable[[CellCompletion], None],
        *,
        max_model_len: int,
        on_output: Callable[[CellTokenOutput], None] | None = None,
        history_store=None,
        storage_enabled=False,
        batch_executor=None,
    ):
        if (
            backend.config.role != "colocated"
            or backend.config.scheduler_pipeline_depth != 1
        ):
            raise NotImplementedError("requires colocated depth-one backend")
        if not callable(on_complete) or not cell_id:
            raise ValueError("completion callback and cell ID required")
        if batch_executor is not None and batch_executor.loop is not loop:
            raise ValueError("Cell and network executor must share the same event loop")
        self.cell_id = cell_id
        self.loop = loop
        self.backend = backend
        self.execution = execution
        self.max_model_len = max_model_len
        self._on_complete = on_complete
        if on_output is not None and not callable(on_output):
            raise TypeError("output callback must be callable")
        self._on_output = on_output
        self._requests = {}
        self._progress = {}
        self._waiting = set()
        self._completed = set()
        self._failed = set()
        self._on_failure = None
        self._kv_bytes = 0
        self._inflight = None
        self._dispatch_pending = False
        self._closed = False
        from cell_storage import SessionHistoryStore

        self.history_store = (
            history_store if history_store is not None else SessionHistoryStore()
        )
        self.storage_enabled = storage_enabled
        self._migration_pause = None
        self._blocked = False
        self.state_records = []
        self.last_planned_step = None
        self.steps = []
        self.load_history = []
        self.communication_batches = []
        self.batch_executor = batch_executor

    @classmethod
    def create(
        cls,
        cell_id,
        loop,
        config,
        on_complete,
        *,
        on_output=None,
        history_store=None,
        batch_executor=None,
    ):
        if config.decode_context_parallel_size > 1:
            from dcp_cell import MidlayerDCPCell

            return MidlayerDCPCell.create(
                cell_id,
                loop,
                config,
                on_complete,
                on_output=on_output,
                history_store=history_store,
                batch_executor=batch_executor,
            )
        if config.data_parallel_size > 1:
            from dp_cell import MidlayerDPCell

            return MidlayerDPCell.create(
                cell_id,
                loop,
                config,
                on_complete,
                on_output=on_output,
                history_store=history_store,
                batch_executor=batch_executor,
            )
        backend, execution = create_midlayer_runtime(config)
        try:
            return cls(
                cell_id,
                loop,
                backend,
                execution,
                on_complete,
                max_model_len=config.max_model_len,
                on_output=on_output,
                history_store=history_store,
                storage_enabled=config.scheduler_adapter == "vllm_0_31",
                batch_executor=batch_executor,
            )
        except BaseException:
            backend.close()
            raise

    def submit_ready_request(self, request: CellRequest):
        if self._closed:
            raise RuntimeError("cell is closed")
        if (
            not isinstance(request, CellRequest)
            or request.cell_ready_time_s != self.loop.now_s
        ):
            raise ValueError("submit only at the actual ready event")
        if request.request_id in self._requests:
            raise ValueError("duplicate cell request ID")
        if request.input_tokens + request.output_tokens > self.max_model_len:
            raise ValueError("request exceeds configured context")
        if request.prefix_hit_tokens and not self.storage_enabled:
            raise NotImplementedError("prefix reuse requires the storage adapter")
        history = self.history_store.prepare(request)
        if history is not None:
            if not self.storage_enabled:
                self.history_store.abort(request)
                raise NotImplementedError(
                    "session caching requires the storage adapter"
                )
            try:
                self.storage_rpc("prepare_history", request.request_id, history)
            except BaseException:
                self.history_store.abort(request)
                raise
        native = RequestSpec(
            request.cell_ready_time_s,
            request.request_id,
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
        )
        try:
            self.backend.add_request(native)
        except BaseException:
            self.history_store.abort(request)
            raise
        self._requests[request.request_id] = request
        self._progress[request.request_id] = {
            "first_execution_s": None,
            "first_token_s": None,
            "last_token_s": None,
            "generated_tokens": 0,
            "prefill_active_s": 0.0,
            "decode_active_s": 0.0,
        }
        self._waiting.add(request.request_id)
        self._record_load()
        self._request_dispatch()

    def _request_dispatch(self):
        if (
            self._inflight is not None
            or self._dispatch_pending
            or self._migration_pause is not None
        ):
            return
        self._dispatch_pending = True
        self.loop.schedule(
            self.loop.now_s, DISPATCH, f"{self.cell_id}:plan", self._dispatch
        )

    def _dispatch(self):
        self._dispatch_pending = False
        if (
            self._inflight is not None
            or self._migration_pause is not None
            or not self.backend.has_work()
        ):
            return
        step = self.backend.plan_step(self.loop.now_s)
        if step is None:
            if not self.storage_enabled:
                raise RuntimeError(
                    "native backend has work but made no causal progress"
                )
            self._blocked = True
            state = self.storage_rpc("storage_snapshot")
            self.state_records.append(
                {
                    "time_s": self.loop.now_s,
                    "state": "waiting_for_resource",
                    "storage": state,
                }
            )
            failures = {
                r["request_id"]: r
                for r in state["records"]
                if r.get("time_s") == self.loop.now_s
                and r["state"] == "capacity_unsatisfiable"
            }
            for rid, reason in failures.items():
                self.storage_rpc("abort_request", rid)
                self._failed.add(rid)
                self._waiting.discard(rid)
                self.history_store.abort(self._requests[rid])
                failure = {
                    "cell_id": self.cell_id,
                    "request_id": rid,
                    "time_s": self.loop.now_s,
                    "reason": "capacity_unsatisfiable",
                    **reason,
                }
                self.state_records.append(failure)
                if self._on_failure is not None:
                    self.loop.schedule(
                        self.loop.now_s,
                        NOTIFY,
                        f"{self.cell_id}:failure:{rid}",
                        lambda failure=failure: self._on_failure(failure),
                    )
            if failures:
                self._blocked = self.backend.has_work()
                self._request_dispatch()
            self._record_load()
            return
        self._blocked = False
        self.last_planned_step = step
        components = self.execution.execute_step(
            step.schedule,
            kv_cache_events=step.kv_cache_events,
        )
        duration = components.execution_timeline.duration_s
        communication = self._communication_batch(step, components)
        if self.batch_executor is not None:
            communication["status"] = "external_network_completion"
            communication["native_call_duration_role"] = "reference_only_not_charged"
        self.communication_batches.append(communication)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid native execution duration")
        start = self.loop.now_s
        end = start + duration if self.batch_executor is None else None
        self._inflight = (step, start, end)
        for request_id in step.schedule.preempted_request_ids:
            self._waiting.add(request_id)
        for scheduled in step.schedule.requests:
            self._waiting.discard(scheduled.request_id)
            progress = self._progress[scheduled.request_id]
            if progress["first_execution_s"] is None:
                progress["first_execution_s"] = start
            if self.batch_executor is None:
                progress[f"{scheduled.phase}_active_s"] += duration
        self._apply_kv(step, releasing=False)
        self._record_load()
        if self.batch_executor is None:
            self.loop.schedule(
                end, STEP_COMPLETE, f"{self.cell_id}:step_done", self._complete_step
            )
        else:
            self.batch_executor.start_batch(
                self.cell_id,
                step.schedule.step_id,
                components,
                self.execution,
                self._network_batch_complete,
            )

    def _network_batch_complete(self, end):
        if (
            self._inflight is None
            or self._inflight[2] is not None
            or end != self.loop.now_s
        ):
            raise RuntimeError("unexpected network batch completion")
        step, start, _ = self._inflight
        self._inflight = (step, start, end)
        for scheduled in step.schedule.requests:
            self._progress[scheduled.request_id][f"{scheduled.phase}_active_s"] += (
                end - start
            )
        self._complete_step()

    def _communication_batch(self, step, components):
        from cell_communication import communication_batch

        return communication_batch(self.cell_id, step.schedule.step_id, components)

    def _apply_kv(self, step, *, releasing):
        if self.storage_enabled:
            # Shared blocks and retained caches cannot be accounted by summing
            # per-request ALLOCATE/RELEASE; the native pool is authoritative.
            self._kv_bytes = self.storage_rpc("storage_snapshot")["referenced_bytes"]
            return
        for event in step.kv_cache_events:
            if event.action == KVCacheAction.ALLOCATE and not releasing:
                self._kv_bytes += event.bytes
            elif event.action == KVCacheAction.RELEASE:
                # Native preemption frees blocks at planning; terminal output
                # frees its blocks only when that step is completed.
                terminal_release = event.request_id in step.finished_request_ids
                if releasing == terminal_release:
                    self._kv_bytes -= event.bytes
            elif event.action in (KVCacheAction.LOAD, KVCacheAction.EVICT):
                raise NotImplementedError(
                    "offload/migration is outside first cell scope"
                )
        capacity = (
            self.backend.config.num_kv_blocks
            * self.backend.config.kv_block_size_tokens
            * self.backend.config.kv_bytes_per_token
        )
        if not 0 <= self._kv_bytes <= capacity:
            raise RuntimeError("KV event accounting exceeds the configured capacity")

    def _complete_step(self):
        if self._inflight is None:
            raise RuntimeError("duplicate step completion")
        step, start, end = self._inflight
        if end != self.loop.now_s:
            raise RuntimeError("step completion at wrong time")
        self.backend.complete_step(step, end)
        self.steps.append(
            {"start_time_s": start, "end_time_s": end, "step": asdict(step)}
        )
        self._apply_kv(step, releasing=True)
        self._inflight = None
        for request_id, count in step.generated_tokens.items():
            progress = self._progress[request_id]
            token_start = progress["generated_tokens"]
            progress["generated_tokens"] += count
            if count:
                if progress["first_token_s"] is None:
                    progress["first_token_s"] = end
                progress["last_token_s"] = end
                if self._on_output is not None:
                    output = CellTokenOutput(
                        self.cell_id,
                        request_id,
                        token_start,
                        count,
                        end,
                        request_id in step.finished_request_ids,
                    )
                    self.loop.schedule(
                        end,
                        NOTIFY,
                        f"{self.cell_id}:output:{request_id}:{token_start}",
                        lambda output=output: self._on_output(output),
                    )
        for request_id in step.finished_request_ids:
            if request_id in self._completed:
                raise RuntimeError("duplicate request completion")
            request = self._requests[request_id]
            progress = self._progress[request_id]
            if progress["generated_tokens"] != request.output_tokens:
                raise RuntimeError(
                    "native completion emitted the wrong number of tokens"
                )
            self._completed.add(request_id)
            self._waiting.discard(request_id)
            self.history_store.complete(request, progress["generated_tokens"])
            result = CellCompletion(
                self.cell_id,
                request_id,
                request.cell_ready_time_s,
                progress["first_execution_s"],
                end,
                progress["first_token_s"],
                progress["last_token_s"],
                progress["generated_tokens"],
                progress["prefill_active_s"],
                progress["decode_active_s"],
            )
            self.loop.schedule(
                end,
                NOTIFY,
                f"{self.cell_id}:request_done:{request_id}",
                lambda result=result: self._on_complete(result),
            )
        self._record_load()
        self._request_dispatch()

    def snapshot(self):
        storage = (
            self.storage_rpc("storage_snapshot")
            if self.storage_enabled and not self._closed
            else None
        )
        if storage is not None:
            self._waiting = set(storage["waiting_request_ids"])
            self._kv_bytes = storage["referenced_bytes"]
        inflight_ids = (
            ()
            if self._inflight is None
            else tuple(item.request_id for item in self._inflight[0].schedule.requests)
        )
        return {
            "cell_id": self.cell_id,
            "time_s": self.loop.now_s,
            "pending_request_count": len(self._requests)
            - len(self._completed)
            - len(self._failed),
            "waiting_request_count": len(self._waiting),
            "scheduled_active_request_count": len(self._requests)
            - len(self._completed)
            - len(self._failed)
            - len(self._waiting),
            "in_flight_request_ids": inflight_ids,
            "kv_allocated_bytes": self._kv_bytes,
            "completed_request_count": len(self._completed),
            "load_source": "adapter_view_of_native_steps_and_kv_events",
            "storage": storage,
            "blocked": self._blocked,
            "migration_paused": self._migration_pause,
            "failed_request_count": len(self._failed),
            "state_records": self.state_records,
            "parallelism": {
                "tensor_parallel_size": self.execution.deployment.worker_pools[
                    "main"
                ].parallel_plan.config.tensor_parallel_size,
                "pipeline_parallel_size": self.execution.deployment.worker_pools[
                    "main"
                ].parallel_plan.config.pipeline_parallel_size,
                "expert_parallel_size": self.execution.deployment.worker_pools[
                    "main"
                ].parallel_plan.config.expert_parallel_size,
                "worker_count": len(
                    self.execution.deployment.worker_pools["main"].ranks
                ),
                "kv_allocated_bytes_scope": "one_rank_uniform_kv_geometry",
            },
            "host_timing": "native_launch_profile"
            if self.execution.host_launch_profiles
            else "device_only",
            "device_communication_timing": "native_static"
            if self.batch_executor is None
            else "external_network_completion",
        }

    def _record_load(self):
        self.load_history.append(self.snapshot())

    def assert_drained(self):
        if self._inflight is not None or self.backend.has_work():
            raise RuntimeError("cell still has native work")
        if (
            len(self._completed) + len(self._failed) != len(self._requests)
            or self._kv_bytes != 0
        ):
            raise RuntimeError("cell has unfinished requests or leaked request KV")

    def storage_rpc(self, name, *args):
        if not self.storage_enabled:
            raise NotImplementedError("native storage RPC requires vLLM 0.31")
        # M-L offers a control-plane injection seam, but not these new RPCs.
        # Access is confined here; the original VLLMBackend is not modified.
        return self.backend._control_plane.rpc(name, *args)

    def set_failure_callback(self, callback):
        if not callable(callback):
            raise TypeError("failure callback must be callable")
        self._on_failure = callback

    def inspect_prefix(self, session_id, context_id, *, input_tokens=None):
        history = self.history_store.history(session_id, context_id)
        return self.storage_rpc("inspect_prefix", history, input_tokens)

    @property
    def next_batch_end_s(self):
        if self._inflight is not None and self._inflight[2] is None:
            raise NotImplementedError(
                "network batch end is unknown until its dependency graph completes"
            )
        return self.loop.now_s if self._inflight is None else self._inflight[2]

    @property
    def storage_signature(self):
        if len(self.execution.deployment.worker_pools["main"].ranks) != 1:
            raise NotImplementedError(
                "static KV migration is not yet connected for multi-GPU Cells"
            )
        model = self.execution.model
        return (
            model.config_sha256,
            self.backend.config.kv_block_size_tokens,
            self.backend.config.kv_bytes_per_token,
            "bf16",
            1,
            1,
        )

    def pause_for_migration(self, transfer_id):
        if self._closed or self._migration_pause is not None:
            raise RuntimeError("Cell already paused/closed")
        self._migration_pause = transfer_id

    def resume_after_migration(self, transfer_id):
        if self._migration_pause != transfer_id:
            raise RuntimeError("migration pause identity mismatch")
        self._migration_pause = None
        self._record_load()
        self._request_dispatch()

    def migration_memory_time(self, size_bytes, access_kind):
        if not size_bytes:
            return 0.0
        hardware = next(iter(self.execution.hardware_by_id.values()))
        bandwidth = hardware.dram_capacity_for_access(
            access_kind,
            active_units=hardware.num_compute_units,
            sm_freq_hz=hardware.nominal_sm_freq_hz,
            dram_freq_hz=hardware.nominal_dram_freq_hz,
        )
        return size_bytes / bandwidth

    def close(self):
        if not self._closed:
            self.backend.close()
            self._closed = True
