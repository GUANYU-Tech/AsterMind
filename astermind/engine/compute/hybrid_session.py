"""Native vLLM 0.30 multi-group session adapters for Fast Mode."""

from dataclasses import asdict, replace
import hashlib
import os

from .native_cell_storage import StorageControlPlane
from src.serving.schedule.schedule_step import BackendStep, ScheduleStep, ScheduledRequest
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent


def start_hybrid_k3_session(native, model, requested, pool, role, budgets, layouts, source, identity,
                            *, export_step_state=False):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.structured_output import StructuredOutputManager
    from src.serving.kv_cache.vllm_allocation import (
        allocate_vllm030_kimi_k3_cache, normalize_vllm030_model_allocation, vllm030_kimi_k3_cache_specs,
    )
    from src.serving.parallelism.parallel_config import ParallelConfig
    from src.serving.schedule.backend_protocol import BackendConfig

    cache = native.cache_config
    specs = vllm030_kimi_k3_cache_specs(native, model)
    state_bytes = max(spec.page_size_bytes for spec in specs.values() if isinstance(spec, MambaSpec))
    mla_bytes = {spec.page_size_bytes // spec.block_size for spec in specs.values() if not isinstance(spec, MambaSpec)}
    per_token, = mla_bytes
    # Match the pinned native hybrid MLA/Mamba page-alignment rule for align mode.
    quantum = max(cache.block_size, 128)
    span = quantum * ((state_bytes + quantum * per_token - 1) // (quantum * per_token))
    cache.block_size = cache.mamba_block_size = span
    cache.mamba_page_size_padded = span * per_token
    cache.enable_prefix_caching = True
    cache.mamba_cache_mode = "align"
    parallel = ParallelConfig(tensor_parallel_size=requested.tensor_parallel_size,
                              pipeline_parallel_size=requested.pipeline_parallel_size,
                              enable_expert_parallel=requested.enable_expert_parallel)
    workers, shared = allocate_vllm030_kimi_k3_cache(native, model, parallel, budgets,
        source_root=source, supported_layouts_by_rank=layouts)
    cache.num_gpu_blocks = shared.num_blocks
    scheduler = Scheduler(native, shared, StructuredOutputManager(native), block_size=span)
    specs = vllm030_kimi_k3_cache_specs(native, model)
    kda, mla, geometry = [], [], []
    for index, group in enumerate(shared.kv_cache_groups):
        # The Scheduler uses one worker's layer names, while PP groups span all stages.
        layer_names = group.layer_names or max(
            (worker.kv_cache_groups[index].layer_names for worker in workers), key=len)
        if not layer_names:
            raise ValueError("K3 native cache group has no source layer on any worker")
        owners = [specs[name] for name in layer_names]
        kinds = {isinstance(spec, MambaSpec) for spec in owners}
        recurrent, = kinds
        (kda if recurrent else mla).append(index)
        sizes = {spec.block_size for spec in owners}
        tokens, = sizes
        geometry.append((tokens, sum(spec.page_size_bytes for spec in owners)))
    if not kda or len(mla) != 1:
        raise ValueError("K3 native cache lost its recurrent or MLA groups")
    kda_layer_groups = {}
    if export_step_state:
        # The Scheduler takes its group layout from one worker. PP workers
        # retain the same group indices but own different source layers.
        for worker in workers:
            if len(worker.kv_cache_groups) != len(geometry):
                raise ValueError("K3 workers disagree with the Scheduler cache group count")
            for index, group in enumerate(worker.kv_cache_groups):
                for name in group.layer_names:
                    spec = specs[name]
                    recurrent = isinstance(spec, MambaSpec)
                    if recurrent != (index in kda) or spec.block_size != geometry[index][0]:
                        raise ValueError("K3 worker cache group disagrees with Scheduler geometry")
                    if recurrent:
                        layer = int(name.split(".")[3])
                        if kda_layer_groups.setdefault(layer, index) != index:
                            raise ValueError("K3 workers disagree on a KDA layer's cache group")
        if set(kda_layer_groups) != {
            layer for layer, kind in enumerate(model.kimi_k3_semantics.layer_attention_types) if kind == "kda"
        }:
            raise ValueError("K3 worker cache groups do not cover every source KDA layer")
    allocation = normalize_vllm030_model_allocation(native, model, tuple(workers), shared, source_root=source)
    cfg = BackendConfig(pool, role, scheduler.max_num_scheduled_tokens,
                       native.scheduler_config.max_num_seqs, shared.num_blocks, span, 0,
                       scheduler_pipeline_depth=1)
    info = dict(version="0.30.0", runtime_acceptance=False, normal_capacity_accepted=False,
                native_allocation=asdict(allocation), k3_source_capacities=(span * shared.num_blocks, shared.num_blocks),
                kv_group_geometry=tuple(geometry), kda_groups=kda, mla_groups=mla,
                shared_cache_arena_bytes_by_rank=tuple(max(t.size for t in w.kv_cache_tensors) for w in workers),
                available_memory_by_rank=budgets, source_identity=identity,
                session_storage=True, effective_prefix_caching=True, mamba_cache_mode="align",
                model_config_sha256=model.config_sha256, source_model_type=model.model_type,
                effective_tensor_parallel_size=native.parallel_config.tensor_parallel_size,
                effective_pipeline_parallel_size=native.parallel_config.pipeline_parallel_size,
                effective_expert_parallel=native.parallel_config.enable_expert_parallel,
                effective_sequence_parallel_moe=native.parallel_config.use_sequence_parallel_moe,
                exports_k3_session_state=export_step_state,
                incremental_session_hash=os.environ.get("ORBITServe_K3_INCREMENTAL_HASH") == "1",
                workers_executed=0, checkpoint_tokens=span, physical_model_execution=False)
    control = HybridK3SessionControl.create(scheduler, cfg,
        replace(requested, enable_prefix_caching=True), info)
    if export_step_state:
        control.kda_layer_groups = tuple(kda_layer_groups.get(layer) for layer in range(model.num_hidden_layers))
    return cfg, control


class NativeGroupedSessionControl(StorageControlPlane):
    def _prefix_hash_block_size(self):
        return self.scheduler.hash_block_size

    @classmethod
    def create(cls, scheduler, config, requested, runtime_info):
        plane = cls.__new__(cls)
        plane.scheduler_config = requested
        plane.runtime_info = runtime_info
        scheduler.orbitserve_capacity = runtime_info["native_allocation"]
        plane._bind(scheduler, config, requested)
        plane.bytes_per_page = max(runtime_info["shared_cache_arena_bytes_by_rank"]) // config.num_kv_blocks
        plane.audit_hits = {}
        return plane

    def add_request(self, spec):
        if spec.prefix_hit_tokens:
            raise ValueError("grouped prefix hits must come from the native allocator")
        super().add_request(spec)

    def blocks(self, request_id):
        return tuple(block for group in self.scheduler.kv_cache_manager.get_blocks(request_id).blocks
                     for block in group if not block.is_null)

    def storage_snapshot(self, compact=False):
        free = self.pool.get_num_free_blocks()
        return dict(referenced_bytes=(self.pool.num_gpu_blocks - 1 - free) * self.bytes_per_page,
                    free_or_reclaimable_blocks=free, records=[] if compact else list(self.records),
                    waiting_request_ids=[rid for rid, req in self.scheduler.requests.items()
                                         if req.status.name != "RUNNING"],
                    hybrid_native_hits={} if compact else dict(self.audit_hits))

    def inspect_prefix(self, history, input_tokens=None):
        from .cell_storage import history_block_hashes

        n = len(history["token_keys"])
        bound = n if input_tokens is None else min(n, max(0, input_tokens - 1))
        hashes = history_block_hashes(history, self._prefix_hash_block_size())
        groups, hit, _ = self.scheduler.kv_cache_manager.coordinator.find_longest_cache_hit(hashes, bound)
        return dict(resident_prefix_tokens=hit, effective_prefix_hit_tokens=hit,
                    block_ids=[[b.block_id for b in group if not b.is_null] for group in groups],
                    bytes=sum(not b.is_null for group in groups for b in group) * self.bytes_per_page,
                    state_version=self.version)

    def plan_step(self, now):
        if self.pending is not None:
            raise RuntimeError("one in-flight batch only")
        if not self.has_work():
            return None
        running = {r.request_id for r in self.scheduler.running}
        native = self.scheduler.schedule()
        if not native.num_scheduled_tokens:
            coordinator = self.scheduler.kv_cache_manager.coordinator
            empty_groups = ((),) * len(coordinator.single_type_managers)
            capacity = self.pool.num_gpu_blocks - 1
            for rid, request in self.scheduler.requests.items():
                if request.is_finished() or request.num_computed_tokens:
                    continue
                needed = coordinator.get_num_blocks_to_allocate(
                    rid, request.num_tokens, empty_groups, 0, 0, 0, request.num_tokens,
                    apply_admission_cap=True)
                state = "capacity_unsatisfiable" if needed > capacity else "waiting_for_capacity_or_resource"
                self._record(rid, state, time_s=now, required_context_blocks=needed,
                             admission_capacity_blocks=capacity)
            return None
        rows, events, generated, finished = [], [], {}, []
        preempted = tuple(sorted(native.preempted_req_ids or ()))
        resumed = native.scheduled_cached_reqs.resumed_req_ids
        for rid, count in native.num_scheduled_tokens.items():
            request = self.scheduler.requests[rid]
            computed = request.num_computed_tokens - count
            spec = self.specs[rid]
            adopted = computed if rid not in running and rid not in resumed else 0
            if adopted:
                self._record_native_hit(rid, request, adopted)
                events.append(KVCacheEvent(rid, KVCacheAction.HIT, "hbm", 0, adopted, 0))
            phase = "prefill" if computed < spec.input_tokens else "decode"
            inputs = min(count, max(0, spec.input_tokens - computed))
            if inputs:
                events.append(KVCacheEvent(rid, KVCacheAction.MISS, "hbm", computed, computed + inputs, 0))
            emit = int(request.num_computed_tokens >= request.num_tokens)
            generated[rid] = emit
            if request.num_output_tokens + emit == spec.output_tokens:
                finished.append(rid)
            high = self.highwater[rid]
            rows.append(ScheduledRequest(rid, phase, inputs, count - inputs,
                computed_tokens=computed, recomputed_tokens=max(0, min(request.num_computed_tokens, high) - computed),
                resumed_from_preemption=rid in resumed, content_prefix_hit_tokens=adopted,
                raw_model_output_tokens=emit))
            self.highwater[rid] = max(high, request.num_computed_tokens)
        rows.sort(key=lambda row: row.phase != "decode")
        step = BackendStep(ScheduleStep(f"{self.config.pool_name}:{self.index}", self.config.pool_name,
                           tuple(rows), tuple(finished), preempted, "mla"), tuple(events), generated,
                           **self._step_metadata(rows))
        self.pending = (step, native)
        self.index += 1
        self.version += 1
        return step

    def _step_metadata(self, rows):
        return {}

    def _record_native_hit(self, rid, request, adopted):
        groups = self.scheduler.kv_cache_manager.get_blocks(rid).blocks
        self.audit_hits[rid] = dict(tokens=adopted, hashed_blocks_after_schedule_by_group=[
            sum(not block.is_null and block.block_hash is not None for block in group)
            for group in groups])

    def complete_step(self, step, now, **flags):
        # Native update_from_output publishes the checkpoints and frees blocks.
        from .latest_vllm_backend import LatestControlPlane

        result = LatestControlPlane.complete_step(self, step, now, **flags)
        for rid in step.finished_request_ids:
            self.specs.pop(rid)
            self.highwater.pop(rid)
            self.histories.pop(rid, None)
        self.version += 1
        return result


class HybridK3SessionControl(NativeGroupedSessionControl):
    def _step_metadata(self, rows):
        if not self.runtime_info["exports_k3_session_state"]:
            return {}
        from src.serving.workload.runtime_inputs import K3SourceInputs

        slots, token_ids = [], []
        recurrent_slots = {group: [] for group in self.runtime_info["kda_groups"]}
        mla_group, = self.runtime_info["mla_groups"]
        geometry = self.runtime_info["kv_group_geometry"]
        for row in rows:
            rid = row.request_id
            request = self.scheduler.requests[rid]
            groups = self.scheduler.kv_cache_manager.get_blocks(rid).blocks
            start, end = row.computed_tokens, row.computed_tokens + row.scheduled_tokens
            token_ids.extend(request.all_token_ids[start:end])
            for group, indices in recurrent_slots.items():
                # vLLM align mode gathers the working state at (seq_len - 1) // block_size.
                span, _ = geometry[group]
                index = (end - 1) // span
                if index >= len(groups[group]) or groups[group][index].is_null:
                    raise ValueError("K3 Scheduler is missing the aligned working KDA state")
                indices.append(groups[group][index].block_id)

            span, _ = geometry[mla_group]
            mla = groups[mla_group]
            for position in range(start, end):
                index, offset = divmod(position, span)
                if index >= len(mla) or mla[index].is_null:
                    raise ValueError("K3 Scheduler is missing a scheduled MLA cache page")
                slots.append(mla[index].block_id * span + offset)

        cache_capacity, state_capacity = self.runtime_info["k3_source_capacities"]
        state = K3SourceInputs(
            request_ids=tuple(row.request_id for row in rows),
            slot_mapping=tuple(slots),
            state_indices_by_layer=tuple(tuple(recurrent_slots[group]) if group is not None else ()
                                         for group in self.kda_layer_groups),
            cache_capacity_tokens=cache_capacity, state_pool_slots=state_capacity,
            prefill_output_source="supports_out" if any(row.phase == "prefill" for row in rows) else None,
        )
        return dict(embedding_token_ids=tuple(token_ids), embedding_token_source="vllm_length_only_simulation",
                    model_source_inputs=state)

    def add_request(self, spec):
        super().add_request(spec)
        if not self.runtime_info["incremental_session_hash"]:
            return
        request = self.scheduler.requests[spec.request_id]
        history = self.histories.get(spec.request_id)
        if history is None:
            namespace = f"request:{spec.request_id}"
            keys = tuple(f"input:{i}" for i in range(spec.input_tokens))
            context = spec.request_id
        else:
            namespace, keys, context = history["session_id"], history["token_keys"], history["context_id"]
        span = self.config.kv_block_size_tokens
        seed = hashlib.sha256(namespace.encode()).digest()
        reference = request._block_hasher
        verify = os.environ.get("ORBITServe_K3_VERIFY_HASH") == "1"

        def hash_new_blocks(native):
            complete = (len(keys) + len(native.output_token_ids)) // span
            count = len(native.block_hashes)
            if complete < count:
                raise ValueError("native symbolic history shrank")
            parent = native.block_hashes[-1] if count else seed
            additions = []
            for block in range(count, complete):
                digest = hashlib.sha256(parent)
                for index in range(block * span, (block + 1) * span):
                    key = keys[index] if index < len(keys) else f"{context}:output:{index - len(keys)}"
                    encoded = key.encode()
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
                parent = digest.digest()
                additions.append(parent)
            if verify and additions != reference(native):
                raise RuntimeError("incremental K3 prefix hash differs from the full chain")
            return additions

        request._block_hasher = hash_new_blocks

    def _record_native_hit(self, rid, request, adopted):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        groups = self.scheduler.kv_cache_manager.get_blocks(rid).blocks
        group_geometry = self.runtime_info["kv_group_geometry"]
        kda, mla = {}, {}
        for indices, target, recurrent in ((self.runtime_info["kda_groups"], kda, True),
                                            (self.runtime_info["mla_groups"], mla, False)):
            for group in indices:
                span, _ = group_geometry[group]
                if adopted % span:
                    raise RuntimeError("hybrid hit is not aligned with every cache group")
                if span != self.scheduler.hash_block_size:
                    raise RuntimeError("K3 prefix hashes must cover the native aligned checkpoint size")
                prefix = groups[group][:adopted // span]
                start = len(prefix) - 1 if recurrent else 0
                if len(prefix) != adopted // span or any(
                    block.is_null or block.block_hash != make_block_hash_with_group_id(request.block_hashes[index], group)
                    for index, block in enumerate(prefix[start:], start)
                ):
                    raise RuntimeError("native hit has no resident KDA checkpoint or MLA pages")
                target[group] = [block.block_id for block in prefix[start:]]
        self.audit_hits[rid] = dict(tokens=adopted, kda_checkpoints=kda, mla_pages=mla)
