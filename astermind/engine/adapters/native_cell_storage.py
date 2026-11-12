"""Stateful adaptation of the real vLLM block pool; M-L source is not patched.

Native schedule/allocate/free/eviction are retained. New code supplies symbolic
session identities, translates real hits, publishes completed blocks, and imports
abstract KV copies into native cache metadata only when transfers have completed.
"""

from dataclasses import replace

from astermind.engine.adapters.cell_storage import history_block_hashes
from astermind.engine.adapters.latest_vllm_backend import LatestControlPlane, construct_scheduler
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduledRequest,
    ScheduleStep,
)


def make_simulation_pd_connector(scheduler, side):
    """Real Scheduler/NIXL control rules, with external simulated completion.

    The NIXL scheduler state is initialized without its socket/thread setup.
    Native matching, allocation metadata and finish retention methods run;
    GPU DMA, leases and heartbeat transport do not run in this CPU adapter.
    """
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
        SupportsHMA,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
        NixlPullConnectorScheduler,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    config = scheduler.vllm_config
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_producer" if side == "producer" else "kv_consumer",
    )

    class SimulationPDConnector(KVConnectorBase_V1, SupportsHMA):
        def __init__(self):
            super().__init__(
                config, KVConnectorRole.SCHEDULER, scheduler.kv_cache_config
            )
            self.side, self.exports = side, {}
            native = NixlPullConnectorScheduler.__new__(NixlPullConnectorScheduler)
            self.native = native
            native.vllm_config, native.kv_cache_config = (
                config,
                scheduler.kv_cache_config,
            )
            native.engine_id = f"simulation:{side}"
            native.side_channel_host, native.side_channel_port = "simulation-only", 0
            native.transfer_tp_size = config.parallel_config.tensor_parallel_size
            groups = scheduler.kv_cache_config.transfer_groups
            native._has_mamba = scheduler.kv_cache_config.has_mamba_layers
            native._is_hma_required = (
                not config.scheduler_config.disable_hybrid_kv_cache_manager
                and any(
                    not isinstance(group.kv_cache_spec, FullAttentionSpec)
                    for group in groups
                )
            )
            native.blocks_per_sw = [
                (
                    (
                        group.kv_cache_spec.sliding_window
                        + group.kv_cache_spec.block_size
                        - 1
                    )
                    // group.kv_cache_spec.block_size
                    + 1
                )
                if isinstance(group.kv_cache_spec, SlidingWindowSpec)
                else 0
                for group in groups
            ]
            native._ssm_spec_blocks = [
                group.kv_cache_spec.num_speculative_blocks
                if isinstance(group.kv_cache_spec, MambaSpec)
                else None
                for group in groups
            ]
            native._ssm_state_slots_are_positional = (
                config.cache_config.mamba_cache_mode == "all"
            )
            native._heartbeat_req_engine, native._heartbeat_by_engine = {}, {}
            native._reqs_need_recv, native._reqs_need_send, native._reqs_need_save = (
                {},
                {},
                {},
            )
            native._reqs_not_processed, native._reqs_in_batch = set(), set()
            native.is_bidirectional_kv_xfer_enabled = native.use_host_buffer = False
            # This lease is native metadata only: no wall-clock expiry is
            # applied to virtual execution. Completion must be explicit.
            native._kv_lease_duration = config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 30
            )
            native.decoder_kv_blocks_ttl = 30

        def on_new_request(self, request):
            # PD parameters are attached by pd_add_request immediately after
            # the native Request is created; that seam calls NIXL on_new_request.
            pass

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return self.native.get_num_new_matched_tokens(request, num_computed_tokens)

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            self.native.update_state_after_alloc(request, blocks, num_external_tokens)

        def build_connector_meta(self, scheduler_output):
            return KVConnectorMetadata()

        def request_finished(self, request, block_ids):
            return self.request_finished_all_groups(request, (list(block_ids),))

        def request_finished_all_groups(self, request, block_ids):
            delayed, params = self.native.request_finished(request, block_ids)
            if delayed:
                self.exports[request.request_id] = params
            return delayed, params

        def update_connector_output(self, output):
            for rid in output.finished_recving or ():
                self.native._stop_heartbeat(rid)
                self.native._reqs_need_recv.pop(rid, None)
            for rid in output.finished_sending or ():
                self.native._reqs_need_send.pop(rid, None)
                self.exports.pop(rid, None)

        def has_pending_block_frees(self):
            return bool(self.exports)

        def start_load_kv(self, *args, **kwargs):
            raise RuntimeError("simulation PD connector has no GPU transport")

        def wait_for_layer_load(self, *args, **kwargs):
            raise RuntimeError("simulation PD connector has no GPU transport")

        def save_kv_layer(self, *args, **kwargs):
            raise RuntimeError("simulation PD connector has no GPU transport")

        def wait_for_save(self):
            raise RuntimeError("simulation PD connector has no GPU transport")

    return SimulationPDConnector()


class StorageControlPlane(LatestControlPlane):
    def __init__(self, config, requested, options):
        self.scheduler = construct_scheduler(config, requested, options)
        self.config = replace(
            config, num_kv_blocks=self.scheduler.orbitserve_capacity["num_kv_blocks"]
        )
        self.requested = requested
        self.specs, self.highwater, self.histories = {}, {}, {}
        self.pending, self.index = None, 0
        self.ready_hashes = {}
        self.exports, self.imports = {}, {}
        self.records = []
        self.version = 0
        self.pd_connector = None

    def configure_pd(self, role, engine_id=None):
        if (
            role not in ("producer", "consumer")
            or self.specs
            or self.pd_connector is not None
        ):
            raise ValueError("configure an empty PD pool once as producer or consumer")
        parallel = self.scheduler.parallel_config
        if self.requested.enable_prefix_caching or parallel.data_parallel_size != 1:
            raise NotImplementedError(
                "PD connector needs DP1 and no APC; TP/PP uses native uniform KV geometry"
            )
        if engine_id is not None and (not isinstance(engine_id, str) or not engine_id):
            raise ValueError("PD engine identity must be nonempty")
        self.pd_connector = make_simulation_pd_connector(self.scheduler, role)
        if engine_id is not None:
            self.pd_connector.native.engine_id = engine_id
        self.scheduler.connector = self.pd_connector
        self.scheduler.requires_kv_delivery = self.pd_connector.requires_kv_delivery
        self.scheduler.connector.bind_kv_cache_manager(self.scheduler.kv_cache_manager)

    def has_work(self):
        if self.pd_connector is None:
            return super().has_work()
        return any(
            not request.is_finished() for request in self.scheduler.requests.values()
        )

    def validate_finished_storage(self, rid):
        if self.pd_connector is None or self.pd_connector.side != "producer":
            return super().validate_finished_storage(rid)
        request = self.scheduler.requests[rid]
        if (
            not request.is_finished()
            or not self.blocks(rid)
            or rid not in self.pd_connector.exports
        ):
            raise RuntimeError("PD producer did not retain native blocks at completion")

    def pd_add_request(self, spec, remote=None):
        if self.pd_connector is None or spec.prefix_hit_tokens:
            raise ValueError(
                "PD needs an explicit connector and no injected prefix hit"
            )
        if self.pd_connector.side == "producer" and (
            remote is not None or spec.output_tokens != 1
        ):
            raise ValueError("NIXL pull producer runs one terminal sample")
        if self.pd_connector.side == "consumer" and (
            remote is None or remote["remote_num_tokens"] != spec.input_tokens
        ):
            raise ValueError("PD consumer needs a completed matching prompt")
        self.add_request(spec)
        request = self.scheduler.requests[spec.request_id]
        request.kv_transfer_params = (
            {"do_remote_decode": True, "do_remote_prefill": False}
            if remote is None
            else dict(remote)
        )
        self.pd_connector.native.on_new_request(request)

    def pd_offer(self, request_id):
        if self.pd_connector is None or self.pd_connector.side != "producer":
            raise ValueError("only a PD producer exports KV")
        return dict(self.pd_connector.exports[request_id])

    def pd_abort_waiting(self, request_id):
        """Cancel before target allocation; acknowledge NIXL's empty cleanup notice."""
        from vllm.v1.outputs import KVConnectorOutput
        from vllm.v1.request import RequestStatus

        if (
            self.pd_connector is None
            or self.pd_connector.side != "consumer"
            or self.pending is not None
            or self.blocks(request_id)
            or request_id in self.pd_connector.native._reqs_need_recv
        ):
            raise ValueError("abort-waiting requires an unallocated PD consumer")
        self.scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
        # Native NIXL emits an empty receive entry to notify the source even
        # though Scheduler already freed this unallocated request. The driver
        # sends that source acknowledgement separately; no receive completion
        # may be fed back to Scheduler for a request it has already removed.
        self.pd_connector.update_connector_output(
            KVConnectorOutput(finished_recving={request_id})
        )
        self.version += 1

    def pd_notify(self, request_id, *, success=True):
        from vllm.v1.outputs import KVConnectorOutput
        from vllm.v1.request import RequestStatus

        if self.pd_connector is None or request_id not in self.scheduler.requests:
            raise ValueError("unknown PD transfer request")
        if self.pd_connector.side == "producer":
            if request_id not in self.pd_connector.exports:
                raise ValueError("PD send completion before source KV is available")
            output = KVConnectorOutput(finished_sending={request_id})
        else:
            if request_id not in self.pd_connector.native._reqs_need_recv:
                raise ValueError("PD receive completion before native allocation")
            if not success:
                self.scheduler.finish_requests(
                    request_id, RequestStatus.FINISHED_ABORTED
                )
            else:
                self.highwater[request_id] = self.scheduler.requests[
                    request_id
                ].num_computed_tokens
            output = KVConnectorOutput(finished_recving={request_id})
        self.scheduler._update_from_kv_xfer_finished(output)
        self.version += 1

    def pd_snapshot(self):
        if self.pd_connector is None:
            raise ValueError("PD connector not configured")
        return {
            "side": self.pd_connector.side,
            "storage": self.storage_snapshot(),
            "requests": {
                rid: {
                    "status": r.status.name,
                    "computed_tokens": r.num_computed_tokens,
                    "block_ids": self.blocks(rid),
                }
                for rid, r in self.scheduler.requests.items()
            },
            "receives": {
                rid: {
                    "block_ids": tuple(tuple(group) for group in state[1]),
                    "waiting": state[3],
                }
                for rid, state in self.pd_connector.native._reqs_need_recv.items()
            },
            "exports": dict(self.pd_connector.exports),
            "transport": "external simulation callbacks; no NIXL GPU worker or lease clock",
        }

    @property
    def pool(self):
        return self.scheduler.kv_cache_manager.block_pool

    @property
    def context_block_size_tokens(self):
        return self.config.kv_block_size_tokens

    def configure_dp_router(self, size):
        """Use the wheel's real DPLB routing method with one local client."""
        from types import SimpleNamespace

        from vllm.v1.engine.core_client import DPLBAsyncMPClient

        if (
            type(size) is not int
            or size < 2
            or size != self.scheduler.parallel_config.data_parallel_size
        ):
            raise ValueError(
                "router size must match the real DP Scheduler configuration"
            )
        if hasattr(self, "dp_router"):
            raise ValueError("DP router already configured")
        self.dp_router_type = DPLBAsyncMPClient
        self.dp_router = SimpleNamespace(
            core_engines=tuple(range(size)),
            lb_engines=[[0, 0, 0.0] for _ in range(size)],
            client_count=1,
            eng_start_index=0,
            engine_inflight=dict.fromkeys(range(size), 0),
            reqs_in_flight={},
        )

    def dp_choose_route(self, request_id, counts):
        from types import SimpleNamespace

        router = self.dp_router
        if (
            not request_id
            or request_id in router.reqs_in_flight
            or len(counts) != len(router.core_engines)
        ):
            raise ValueError("invalid DP routing request")
        for waiting, running, usage in counts:
            if (
                type(waiting) is not int
                or type(running) is not int
                or min(waiting, running) < 0
                or not 0 <= usage <= 1
            ):
                raise ValueError("invalid native DP load counts")
        router.lb_engines = [list(row) for row in counts]
        request = SimpleNamespace(
            request_id=request_id, data_parallel_rank=None, pooling_params=None
        )
        return self.dp_router_type.get_core_engine_for_request(router, request)

    def dp_finish_routes(self, request_ids):
        import asyncio
        from types import SimpleNamespace

        if any(rid not in self.dp_router.reqs_in_flight for rid in request_ids):
            raise ValueError("cannot complete an unknown DP route")
        asyncio.run(
            self.dp_router_type.process_engine_outputs(
                self.dp_router, SimpleNamespace(finished_requests=set(request_ids))
            )
        )

    def dp_router_snapshot(self):
        return {
            "source": "vllm.v1.engine.core_client.DPLBAsyncMPClient",
            "client_count": self.dp_router.client_count,
            "inflight_by_engine": dict(self.dp_router.engine_inflight),
            "request_owners": dict(self.dp_router.reqs_in_flight),
        }

    def prepare_history(self, request_id, history):
        if request_id in self.histories or request_id in self.specs:
            raise ValueError("duplicate request history")
        self.histories[request_id] = history

    def add_request(self, spec):
        from vllm import SamplingParams
        from vllm.v1.request import Request

        if spec.request_id in self.specs or spec.output_tokens < 1:
            raise ValueError("unique positive-output requests required")
        history = self.histories.get(spec.request_id)
        if spec.prefix_hit_tokens and history is None:
            raise ValueError("a claimed hit without a session identity is unsupported")
        if history is not None and not self.requested.enable_prefix_caching:
            raise ValueError("session reuse requires enable_prefix_caching")

        def hasher(request):
            identity = history or {
                "session_id": f"request:{spec.request_id}",
                "context_id": spec.request_id,
                "token_keys": tuple(f"input:{i}" for i in range(spec.input_tokens)),
            }
            keys = identity["token_keys"] + tuple(
                f"{identity['context_id']}:output:{i}"
                for i in range(len(request.output_token_ids))
            )
            full = history_block_hashes(
                {**identity, "token_keys": keys}, self.config.kv_block_size_tokens
            )
            return full[len(request.block_hashes) :]

        request = Request(
            spec.request_id,
            [0] * spec.input_tokens,
            SamplingParams(max_tokens=spec.output_tokens, ignore_eos=True),
            None,
            arrival_time=spec.arrival_time_s,
            block_hasher=hasher if self.requested.enable_prefix_caching else None,
            session_id=None if history is None else history["session_id"],
        )
        self.scheduler.add_request(request)
        self.specs[spec.request_id] = spec
        self.highwater[spec.request_id] = 0
        self.version += 1

    def _record(self, rid, state, **extra):
        request = self.scheduler.requests.get(rid)
        self.records.append(
            {
                "request_id": rid,
                "state": state,
                "known_context_tokens": None if request is None else request.num_tokens,
                "free_blocks": self.pool.get_num_free_blocks(),
                "watermark_blocks": self.scheduler.kv_cache_manager.watermark_blocks,
                **extra,
            }
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
        events, preempted = [], []
        for rid, (computed, status, blocks, _) in before.items():
            r = self.scheduler.requests[rid]
            if status != "PREEMPTED" and r.status.name == "PREEMPTED":
                preempted.append(rid)
                self._record(rid, "preempted_recompute", time_s=now)
                if blocks:
                    events.append(
                        self.kv_event(rid, KVCacheAction.RELEASE, 0, len(blocks))
                    )
        if not native.num_scheduled_tokens:
            self.version += 1
            for rid, r in self.scheduler.requests.items():
                if r.is_finished():
                    continue
                needed = (
                    r.num_tokens + self.context_block_size_tokens - 1
                ) // self.context_block_size_tokens
                admission_limit = self.pool.num_gpu_blocks - 1
                reason = (
                    "waiting_for_remote_kv"
                    if r.status.name == "WAITING_FOR_REMOTE_KVS"
                    else (
                        "capacity_unsatisfiable"
                        if needed > admission_limit
                        else "waiting_for_capacity_or_resource"
                    )
                )
                self._record(
                    rid,
                    reason,
                    time_s=now,
                    required_context_blocks=needed,
                    admission_capacity_blocks=admission_limit,
                )
            return None
        scheduled, generated, finished = [], {}, []
        for rid, count in native.num_scheduled_tokens.items():
            old_computed, status, old_blocks, generation_boundary = before[rid]
            r = self.scheduler.requests[rid]
            computed = r.num_computed_tokens - count
            adopted = max(0, computed - old_computed)
            phase = "prefill" if computed < self.specs[rid].input_tokens else "decode"
            emit = int(r.num_computed_tokens >= generation_boundary)
            recomputed = max(
                0, min(r.num_computed_tokens, self.highwater[rid]) - computed
            )
            input_count = (
                min(count, max(0, self.specs[rid].input_tokens - computed))
                if phase == "prefill"
                else 0
            )
            trace_hit = max(self.specs[rid].prefix_hit_tokens, adopted)
            scheduled.append(
                ScheduledRequest(
                    rid,
                    phase,
                    input_count,
                    count - input_count,
                    computed_tokens=computed,
                    recomputed_tokens=recomputed,
                    resumed_from_preemption=status == "PREEMPTED",
                    trace_prefix_hit_tokens=trace_hit,
                    scheduler_prefix_hit_tokens=adopted,
                    raw_model_output_tokens=emit,
                )
            )
            generated[rid] = emit
            new_blocks = self.blocks(rid)
            old_count = len(old_blocks) + adopted // self.config.kv_block_size_tokens
            if len(new_blocks) > old_count:
                events.append(
                    self.kv_event(
                        rid, KVCacheAction.ALLOCATE, old_count, len(new_blocks)
                    )
                )
            if adopted:
                events.append(
                    KVCacheEvent(
                        rid,
                        KVCacheAction.HIT,
                        "hbm",
                        0,
                        adopted,
                        adopted * self.config.kv_bytes_per_token,
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
            self._record(
                rid,
                "scheduled",
                time_s=now,
                effective_prefix_hit_tokens=adopted,
                requested_new_blocks=max(0, len(new_blocks) - old_count),
                referenced_block_ids=list(new_blocks),
            )
            if len(r.output_token_ids) + emit == self.specs[rid].output_tokens:
                finished.append(rid)
                if new_blocks and (
                    self.pd_connector is None or self.pd_connector.side != "producer"
                ):
                    events.append(
                        self.kv_event(rid, KVCacheAction.RELEASE, 0, len(new_blocks))
                    )
        for rid in set(before) - set(native.num_scheduled_tokens):
            request = self.scheduler.requests[rid]
            state = (
                "pd_retained"
                if self.pd_connector is not None and request.is_finished()
                else (
                    "waiting"
                    if request.status.name != "RUNNING"
                    else "running_not_scheduled"
                )
            )
            self._record(rid, state, time_s=now)
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
        self.pending = (step, native)
        self.index += 1
        self.version += 1
        return step

    def complete_step(self, step, now, **flags):
        # Publication is fenced by actual simulated completion, before native free.
        for rid in step.generated_tokens:
            r = self.scheduler.requests[rid]
            for i, block_id in enumerate(self.blocks(rid)):
                if (i + 1) * self.config.kv_block_size_tokens <= r.num_computed_tokens:
                    block = self.pool.blocks[block_id]
                    if block.block_hash is not None:
                        self.ready_hashes[block_id] = block.block_hash
        result = super().complete_step(step, now, **flags)
        self.version += 1
        return result

    def _ready_block(self, digest):
        blocks = self.pool.get_cached_block(digest, [0])
        if not blocks:
            return None
        block = blocks[0]
        return (
            block if self.ready_hashes.get(block.block_id) == block.block_hash else None
        )

    def inspect_prefix(self, history, input_tokens=None):
        hashes, ids = (
            history_block_hashes(history, self.config.kv_block_size_tokens),
            [],
        )
        for digest in hashes:
            block = self._ready_block(digest)
            if block is None:
                break
            ids.append(block.block_id)
        resident = len(ids) * self.config.kv_block_size_tokens
        effective = (
            resident
            if input_tokens is None
            else min(
                resident,
                max(0, input_tokens - 1)
                // self.config.kv_block_size_tokens
                * self.config.kv_block_size_tokens,
            )
        )
        return {
            "resident_prefix_tokens": resident,
            "effective_prefix_hit_tokens": effective,
            "block_ids": ids,
            "bytes": resident * self.config.kv_bytes_per_token,
            "state_version": self.version,
        }

    def storage_snapshot(self):
        blocks = [b for b in self.pool.blocks if not b.is_null]
        referenced = sum(b.ref_cnt > 0 for b in blocks)
        cached = sum(
            b.block_hash is not None
            and self.ready_hashes.get(b.block_id) == b.block_hash
            for b in blocks
        )
        evictable_cached = sum(
            b.ref_cnt == 0
            and b.block_hash is not None
            and self.ready_hashes.get(b.block_id) == b.block_hash
            for b in blocks
        )
        size = self.config.kv_block_size_tokens * self.config.kv_bytes_per_token
        return {
            "source": "native_vllm_block_pool",
            "state_version": self.version,
            "capacity": self.scheduler.orbitserve_capacity,
            "total_blocks": self.pool.num_gpu_blocks,
            "usable_blocks": len(blocks),
            "referenced_unique_blocks": referenced,
            "referenced_bytes": referenced * size,
            "free_or_reclaimable_blocks": self.pool.get_num_free_blocks(),
            "kv_cache_usage": self.scheduler.kv_cache_manager.usage,
            "ready_cached_blocks": cached,
            "evictable_cached_blocks": evictable_cached,
            "migration_reserved_blocks": sum(
                len(t["new"]) for t in self.imports.values()
            ),
            "waiting_request_ids": [
                rid
                for rid, r in self.scheduler.requests.items()
                if r.status.name != "RUNNING"
            ],
            "running_request_ids": [
                rid
                for rid, r in self.scheduler.requests.items()
                if r.status.name == "RUNNING"
            ],
            "records": list(self.records),
        }

    def abort_request(self, request_id):
        from vllm.v1.request import RequestStatus

        if self.pending is not None:
            raise RuntimeError("cannot abort while a batch is executing")
        self.scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
        self.version += 1

    def export_prefix(self, transfer_id, history, mode="copy"):
        if transfer_id in self.exports:
            return self.exports[transfer_id]["payload"]
        hashes, blocks = [], []
        for digest in history_block_hashes(history, self.config.kv_block_size_tokens):
            block = self._ready_block(digest)
            if block is None:
                break
            if mode == "move" and block.ref_cnt:
                raise ValueError("move requires an unreferenced source prefix")
            hashes.append(digest)
            blocks.append(block)
        if not blocks:
            raise ValueError("source has no ready complete prefix")
        self.pool.touch(blocks)
        payload = {
            "hashes": tuple(hashes),
            "block_bytes": self.config.kv_block_size_tokens
            * self.config.kv_bytes_per_token,
            "history": history,
            "source_block_ids": tuple(b.block_id for b in blocks),
        }
        self.exports[transfer_id] = {"payload": payload, "blocks": blocks}
        self.version += 1
        return payload

    def reserve_import(self, transfer_id, payload):
        if transfer_id in self.imports:
            return self.imports[transfer_id]["result"]
        existing, missing = [], []
        for digest in payload["hashes"]:
            block = self._ready_block(digest)
            if block is None:
                missing.append(digest)
            else:
                existing.append(block)
        self.pool.touch(existing)
        if (
            missing
            and len(missing) + self.scheduler.kv_cache_manager.watermark_blocks
            > self.pool.get_num_free_blocks()
        ):
            self.pool.free_blocks(existing)
            return {
                "accepted": False,
                "reason": "target_capacity_temporarily_unavailable",
            }
        new = self.pool.get_new_blocks(len(missing))
        result = {
            "accepted": True,
            "missing_blocks": len(new),
            "bytes": len(new) * payload["block_bytes"],
        }
        self.imports[transfer_id] = {
            "existing": existing,
            "new": new,
            "hashes": missing,
            "result": result,
        }
        self.version += 1
        return result

    def commit_import(self, transfer_id):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        state = self.imports.pop(transfer_id)
        for digest, block in zip(state["hashes"], state["new"], strict=True):
            # Version-specific adapter seam: attach imported symbolic data to native APC.
            self.pool._insert_block_hash(
                make_block_hash_with_group_id(digest, 0),
                block,
                self.config.kv_block_size_tokens,
            )
            self.ready_hashes[block.block_id] = block.block_hash
        self.pool.free_blocks(state["new"] + state["existing"])
        self.version += 1

    def rollback_import(self, transfer_id):
        state = self.imports.pop(transfer_id, None)
        if state is not None:
            self.pool.free_blocks(state["new"] + state["existing"])
            self.version += 1

    def release_export(self, transfer_id, move=False):
        state = self.exports.pop(transfer_id, None)
        if state is None:
            return {"removed_blocks": 0, "retained_blocks": 0}
        removable = [b for b in state["blocks"] if move and b.ref_cnt == 1]
        if removable:
            self.pool.evict_blocks({b.block_id for b in removable})
        self.pool.free_blocks(state["blocks"])
        self.version += 1
        return {
            "removed_blocks": len(removable),
            "retained_blocks": len(state["blocks"]) - len(removable),
        }
