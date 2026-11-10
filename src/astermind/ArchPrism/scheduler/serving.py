"""Drive dev's Scheduler and publish request progress for one Cell."""

from dataclasses import asdict
from typing import Callable

from orbitserve.runtime.engine.cell_events import (
    CellCompletion, CellEventLoop, CellRequest, CellTokenOutput, DISPATCH, NOTIFY,
)
from src.serving.kv_cache.kv_cache_events import KVCacheAction
from src.serving.request_gen.request_stream import RequestSpec

from ...cell_storage import SessionHistoryStore
from ..midlayer.runtime import create_midlayer_runtime, prepare_scheduled_step


class ServingScheduler:
    def __init__(self, cell_id: str, loop: CellEventLoop, backend, execution,
                 on_complete: Callable[[CellCompletion], None], *, max_model_len: int,
                 on_output: Callable[[CellTokenOutput], None] | None = None,
                 on_failure: Callable[[dict], None] | None = None,
                 history_store=None, storage_enabled=False, batch_executor):
        if backend.config.scheduler_pipeline_depth != 1:
            raise NotImplementedError("requires a depth-one backend")
        if not callable(on_complete) or not cell_id:
            raise ValueError("completion callback and cell ID required")
        if batch_executor.loop is not loop:
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
        if on_failure is not None and not callable(on_failure):
            raise TypeError("failure callback must be callable")
        self._on_failure = on_failure
        self._kv_bytes = 0
        self._inflight = None
        self._dispatch_pending = False
        self._closed = False
        self.history_store = history_store if history_store is not None else SessionHistoryStore()
        self.storage_enabled = storage_enabled
        self._migration_pause = None
        self._blocked = False
        self.state_records = []
        self.steps = []
        self.load_history = []
        self.batch_executor = batch_executor

    @classmethod
    def create(cls, cell_id, loop, config, on_complete, *, on_output=None, on_failure=None,
               history_store=None, batch_executor):
        backends, execution, storage_enabled = create_midlayer_runtime(config)
        backend = backends["main"]
        try:
            return cls(cell_id, loop, backend, execution, on_complete,
                       max_model_len=config.max_model_len, on_output=on_output,
                       on_failure=on_failure,
                       history_store=history_store, storage_enabled=storage_enabled,
                       batch_executor=batch_executor)
        except BaseException:
            backend.close()
            raise

    def submit_ready_request(self, request: CellRequest):
        if self._closed:
            raise RuntimeError("cell is closed")
        if not isinstance(request, CellRequest) or request.cell_ready_time_s != self.loop.now_s:
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
                raise NotImplementedError("session caching requires the storage adapter")
            try:
                self.storage_rpc("prepare_history", request.request_id, history)
            except BaseException:
                self.history_store.abort(request)
                raise
        native = RequestSpec(request.cell_ready_time_s, request.request_id,
                             request.input_tokens, request.output_tokens, request.prefix_hit_tokens)
        try:
            self.backend.add_request(native)
        except BaseException:
            self.history_store.abort(request)
            raise
        self._requests[request.request_id] = request
        self._progress[request.request_id] = {
            "first_execution_s": None, "first_token_s": None, "last_token_s": None,
            "generated_tokens": 0, "prefill_active_s": 0.0, "decode_active_s": 0.0,
        }
        self._waiting.add(request.request_id)
        self._record_load()
        self._request_dispatch()

    def _request_dispatch(self):
        if self._inflight is not None or self._dispatch_pending or self._migration_pause is not None:
            return
        self._dispatch_pending = True
        self.loop.schedule(self.loop.now_s, DISPATCH, "batch:dispatch", self._dispatch,
                           owner_id=self.cell_id, metadata={"pool_name": self.backend.config.pool_name})

    def _dispatch(self):
        self._dispatch_pending = False
        if self._inflight is not None or self._migration_pause is not None or not self.backend.has_work():
            return
        step = self.backend.plan_step(self.loop.now_s)
        if step is None:
            if not self.storage_enabled:
                raise RuntimeError("native backend has work but made no causal progress")
            self._blocked = True
            state = self.storage_rpc("storage_snapshot")
            self.state_records.append({"time_s": self.loop.now_s, "state": "waiting_for_resource", "storage": state})
            failures = {r["request_id"]: r for r in state["records"]
                        if r.get("time_s") == self.loop.now_s and r["state"] == "capacity_unsatisfiable"}
            for rid, reason in failures.items():
                self.storage_rpc("abort_request", rid)
                self._failed.add(rid)
                self._waiting.discard(rid)
                self.history_store.abort(self._requests[rid])
                failure = {"cell_id": self.cell_id, "request_id": rid, "time_s": self.loop.now_s,
                           "reason": "capacity_unsatisfiable", **reason}
                self.state_records.append(failure)
                if self._on_failure is not None:
                    self.loop.schedule(self.loop.now_s, NOTIFY, f"{self.cell_id}:failure:{rid}",
                                       lambda failure=failure: self._on_failure(failure))
            if failures:
                self._blocked = self.backend.has_work()
                self._request_dispatch()
            self._record_load()
            return
        self._blocked = False
        plan = prepare_scheduled_step(
            self.execution, step,
            source_prompt_lengths=tuple(self._requests[row.request_id].input_tokens
                                        for row in step.schedule.requests),
        )
        start = self.loop.now_s
        self._inflight = (step, start)
        for request_id in step.schedule.preempted_request_ids:
            self._waiting.add(request_id)
        for scheduled in step.schedule.requests:
            self._waiting.discard(scheduled.request_id)
            progress = self._progress[scheduled.request_id]
            if progress["first_execution_s"] is None:
                progress["first_execution_s"] = start
        self._apply_kv(step, releasing=False)
        self._record_load()
        self.batch_executor.start_batch(self.cell_id, step.schedule.step_id, plan,
                                       self.execution, self._complete_step,
                                       token_count=step.schedule.token_count, pool_name=step.schedule.pool_name,
                                       on_next_batch=self._continue_batch)

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
                raise NotImplementedError("offload/migration is outside first cell scope")
        capacity = (self.backend.config.num_kv_blocks * self.backend.config.kv_block_size_tokens
                    * self.backend.config.kv_bytes_per_token)
        if not 0 <= self._kv_bytes <= capacity:
            raise RuntimeError("KV event accounting exceeds the configured capacity")

    def _complete_step(self, end):
        if self._inflight is None:
            raise RuntimeError("duplicate step completion")
        step, start = self._inflight
        if end != self.loop.now_s:
            raise RuntimeError("step completion at wrong time")
        for scheduled in step.schedule.requests:
            self._progress[scheduled.request_id][f"{scheduled.phase}_active_s"] += end - start
        self.backend.complete_step(step, end)
        self.steps.append({"start_time_s": start, "end_time_s": end,
                           "step": asdict(step)})
        self._apply_kv(step, releasing=True)
        self._inflight = None
        outputs = []
        for request_id, count in step.generated_tokens.items():
            progress = self._progress[request_id]
            token_start = progress["generated_tokens"]
            progress["generated_tokens"] += count
            if count:
                if progress["first_token_s"] is None:
                    progress["first_token_s"] = end
                progress["last_token_s"] = end
                outputs.append(CellTokenOutput(
                    self.cell_id, request_id, token_start, count, end,
                    request_id in step.finished_request_ids,
                ))
        completions = []
        for request_id in step.finished_request_ids:
            if request_id in self._completed:
                raise RuntimeError("duplicate request completion")
            request = self._requests[request_id]
            progress = self._progress[request_id]
            expected = request.output_tokens
            if self.backend.config.is_prefill_only:
                expected = min(1, expected)
            elif self.backend.config.is_decode_only:
                expected -= 1  # Prefill has already produced the first token.
            if progress["generated_tokens"] != expected:
                raise RuntimeError("native completion emitted the wrong number of tokens")
            self._completed.add(request_id)
            self._waiting.discard(request_id)
            self.history_store.complete(request, progress["generated_tokens"])
            completions.append(CellCompletion(
                self.cell_id, request_id, request.cell_ready_time_s, progress["first_execution_s"], end,
                progress["first_token_s"], progress["last_token_s"], progress["generated_tokens"],
                progress["prefill_active_s"], progress["decode_active_s"],
            ))

        self._record_load()
        if self._on_output is not None:
            for output in outputs:
                self._on_output(output)
        for completion in completions:
            self._on_complete(completion)

    def _continue_batch(self):
        if not self._dispatch_pending and len(self._requests) > len(self._completed) + len(self._failed):
            self._dispatch()

    def snapshot(self):
        pool = self.execution.deployment.worker_pools[self.backend.config.pool_name]
        storage = self.storage_rpc("storage_snapshot") if self.storage_enabled and not self._closed else None
        if storage is not None:
            self._waiting = set(storage["waiting_request_ids"])
            self._kv_bytes = storage["referenced_bytes"]
        inflight_ids = () if self._inflight is None else tuple(
            item.request_id for item in self._inflight[0].schedule.requests)
        return {
            "cell_id": self.cell_id, "pool_name": pool.name, "time_s": self.loop.now_s,
            "pending_request_count": len(self._requests) - len(self._completed) - len(self._failed),
            "waiting_request_count": len(self._waiting),
            "scheduled_active_request_count": len(self._requests) - len(self._completed) - len(self._failed) - len(self._waiting),
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
                "tensor_parallel_size": pool.parallel_plan.config.tensor_parallel_size,
                "pipeline_parallel_size": pool.parallel_plan.config.pipeline_parallel_size,
                "expert_parallel_size": pool.parallel_plan.config.expert_parallel_size,
                "worker_count": len(pool.ranks),
                "kv_allocated_bytes_scope": "representative_rank",
            },
            "host_timing": "native_launch_profile" if self.execution.host_launch_profiles else "device_only",
            "device_communication_timing": "isolated_operator_estimate",
        }

    def report(self):
        runtime = self.backend.runtime_info
        return dict(steps=self.steps, load_history=self.load_history, final_load=self.snapshot(),
                    execution_performance=self.execution.performance_stats(),
                    scheduler_runtime=runtime if isinstance(runtime, dict) else asdict(runtime))

    def _record_load(self):
        self.load_history.append(self.snapshot())

    def storage_rpc(self, name, *args):
        if not self.storage_enabled:
            raise NotImplementedError("native storage RPC requires vLLM 0.31")
        # M-L offers a control-plane injection seam, but not these new RPCs.
        # Access is confined here; the original VLLMBackend is not modified.
        return self.backend._control_plane.rpc(name, *args)

    def inspect_prefix(self, session_id, context_id, *, input_tokens=None):
        history = self.history_store.history(session_id, context_id)
        return self.storage_rpc("inspect_prefix", history, input_tokens)

    @property
    def storage_signature(self):
        """Require identical logical block and per-rank TP/PP partitions.

        One native allocator manages logical blocks across all worker ranks.
        NetworkKVMigration transfers each physical rank's bytes before commit.
        This does not implement redistribution between different partitions.
        """
        if not self.storage_enabled:
            raise NotImplementedError("session migration requires the native storage adapter")
        pool = self.execution.deployment.worker_pools["main"]
        plan = pool.parallel_plan.config
        geometry = self.backend.runtime_info.capacity["rank_geometry"]
        layout = tuple(sorted((row["rank"], row["layer_start"], row["layer_end"],
                               row["kv_block_bytes"]) for row in geometry))
        if {row[0] for row in layout} != set(pool.ranks):
            raise ValueError("native KV geometry must cover every physical rank")
        if len({row[3] for row in layout}) != 1:
            raise NotImplementedError("migration needs native uniform per-rank KV block sizes")
        return (self.execution.model.config_sha256, self.backend.config.kv_block_size_tokens,
                self.backend.config.kv_bytes_per_token, "bf16", plan.tensor_parallel_size,
                plan.pipeline_parallel_size, plan.expert_parallel_size, layout)

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
            access_kind, active_units=hardware.num_compute_units,
            sm_freq_hz=hardware.nominal_sm_freq_hz, dram_freq_hz=hardware.nominal_dram_freq_hz)
        return size_bytes / bandwidth

    def close(self):
        if not self._closed:
            self.backend.close()
            self._closed = True
