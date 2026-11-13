"""Stateful adaptation of the real vLLM block pool.

Native schedule/allocate/free/eviction are retained. New code supplies symbolic
session identities, translates real hits, publishes completed blocks, and imports
abstract KV copies into native cache metadata only when transfers have completed.
"""
from dataclasses import replace

from .latest_vllm_backend import LatestControlPlane, construct_scheduler
from .cell_storage import history_block_hashes
from src.serving.schedule.schedule_step import BackendStep, ScheduleStep, ScheduledRequest
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent


class StorageControlPlane(LatestControlPlane):
    def __init__(self, config, requested, options):
        self.scheduler = construct_scheduler(config, requested, options)
        self.config = replace(config, num_kv_blocks=self.scheduler.orbitserve_capacity["num_kv_blocks"])
        self.requested = requested
        self.specs, self.highwater, self.histories = {}, {}, {}
        self.pending, self.index = None, 0
        self.ready_hashes = {}
        self.exports, self.imports = {}, {}
        self.records = []
        self.version = 0

    @property
    def pool(self):
        return self.scheduler.kv_cache_manager.block_pool

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
            identity = history or {"session_id": f"request:{spec.request_id}",
                                   "context_id": spec.request_id,
                                   "token_keys": tuple(f"input:{i}" for i in range(spec.input_tokens))}
            keys = identity["token_keys"] + tuple(
                f"{identity['context_id']}:output:{i}" for i in range(len(request.output_token_ids)))
            full = history_block_hashes({**identity, "token_keys": keys}, self.config.kv_block_size_tokens)
            return full[len(request.block_hashes):]
        request = Request(spec.request_id, [0] * spec.input_tokens,
                          SamplingParams(max_tokens=spec.output_tokens, ignore_eos=True), None,
                          arrival_time=spec.arrival_time_s,
                          block_hasher=hasher if self.requested.enable_prefix_caching else None,
                          session_id=None if history is None else history["session_id"])
        self.scheduler.add_request(request)
        self.specs[spec.request_id] = spec
        self.highwater[spec.request_id] = 0
        self.version += 1

    def _record(self, rid, state, **extra):
        request = self.scheduler.requests.get(rid)
        self.records.append({"request_id": rid, "state": state,
                             "known_context_tokens": None if request is None else request.num_tokens,
                             "free_blocks": self.pool.get_num_free_blocks(),
                             "watermark_blocks": self.scheduler.kv_cache_manager.watermark_blocks,
                             **extra})

    def plan_step(self, now):
        if self.pending is not None:
            raise RuntimeError("one in-flight batch only")
        if not self.has_work():
            return None
        before = {rid: (r.num_computed_tokens, r.status.name, self.blocks(rid), r.num_tokens)
                  for rid, r in self.scheduler.requests.items()}
        native = self.scheduler.schedule()
        events, preempted = [], []
        for rid, (computed, status, blocks, _) in before.items():
            r = self.scheduler.requests[rid]
            if status != "PREEMPTED" and r.status.name == "PREEMPTED":
                preempted.append(rid)
                self._record(rid, "preempted_recompute", time_s=now)
                if blocks:
                    events.append(self.kv_event(rid, KVCacheAction.RELEASE, 0, len(blocks)))
        if not native.num_scheduled_tokens:
            self.version += 1
            for rid, r in self.scheduler.requests.items():
                if r.is_finished():
                    continue
                needed = (r.num_tokens + self.config.kv_block_size_tokens - 1) // self.config.kv_block_size_tokens
                admission_limit = self.pool.num_gpu_blocks - 1
                reason = "capacity_unsatisfiable" if needed > admission_limit else "waiting_for_capacity_or_resource"
                self._record(rid, reason, time_s=now, required_context_blocks=needed,
                             admission_capacity_blocks=admission_limit)
            return None
        scheduled, generated, finished, token_ids = [], {}, [], []
        for rid, count in native.num_scheduled_tokens.items():
            old_computed, status, old_blocks, generation_boundary = before[rid]
            r = self.scheduler.requests[rid]
            computed = r.num_computed_tokens - count
            token_ids.extend(r.all_token_ids[computed:computed + count])
            adopted = max(0, computed - old_computed)
            phase = "prefill" if computed < self.specs[rid].input_tokens else "decode"
            emit = int(r.num_computed_tokens >= generation_boundary)
            recomputed = max(0, min(r.num_computed_tokens, self.highwater[rid]) - computed)
            input_count = min(count, max(0, self.specs[rid].input_tokens - computed)) if phase == "prefill" else 0
            trace_hit = max(self.specs[rid].prefix_hit_tokens, adopted)
            scheduled.append(ScheduledRequest(rid, phase, input_count, count - input_count,
                             computed_tokens=computed, recomputed_tokens=recomputed,
                             resumed_from_preemption=status == "PREEMPTED",
                             trace_prefix_hit_tokens=trace_hit, scheduler_prefix_hit_tokens=adopted,
                             raw_model_output_tokens=emit))
            generated[rid] = emit
            new_blocks = self.blocks(rid)
            old_count = len(old_blocks) + adopted // self.config.kv_block_size_tokens
            if len(new_blocks) > old_count:
                events.append(self.kv_event(rid, KVCacheAction.ALLOCATE, old_count, len(new_blocks)))
            if adopted:
                events.append(KVCacheEvent(rid, KVCacheAction.HIT, "hbm", 0, adopted,
                                           adopted * self.config.kv_bytes_per_token))
            if input_count:
                events.append(KVCacheEvent(rid, KVCacheAction.MISS, "hbm", computed,
                                           computed + input_count, input_count * self.config.kv_bytes_per_token))
            self.highwater[rid] = max(self.highwater[rid], r.num_computed_tokens)
            self._record(rid, "scheduled", time_s=now, effective_prefix_hit_tokens=adopted,
                         requested_new_blocks=max(0, len(new_blocks) - old_count),
                         referenced_block_ids=list(new_blocks))
            if len(r.output_token_ids) + emit == self.specs[rid].output_tokens:
                finished.append(rid)
                if new_blocks:
                    events.append(self.kv_event(rid, KVCacheAction.RELEASE, 0, len(new_blocks)))
        for rid in set(before) - set(native.num_scheduled_tokens):
            request = self.scheduler.requests[rid]
            state = "waiting" if request.status.name != "RUNNING" else "running_not_scheduled"
            self._record(rid, state, time_s=now)
        step = BackendStep(ScheduleStep(f"{self.config.pool_name}:{self.index}", self.config.pool_name,
                           tuple(scheduled), tuple(finished), tuple(preempted), "paged_varlen"), tuple(events), generated,
                           embedding_token_ids=tuple(token_ids), embedding_token_source="vllm_length_only_simulation")
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
        return block if self.ready_hashes.get(block.block_id) == block.block_hash else None

    def inspect_prefix(self, history, input_tokens=None):
        hashes, ids = history_block_hashes(history, self.config.kv_block_size_tokens), []
        for digest in hashes:
            block = self._ready_block(digest)
            if block is None:
                break
            ids.append(block.block_id)
        resident = len(ids) * self.config.kv_block_size_tokens
        effective = resident if input_tokens is None else min(resident, max(0, input_tokens - 1) // self.config.kv_block_size_tokens * self.config.kv_block_size_tokens)
        return {"resident_prefix_tokens": resident, "effective_prefix_hit_tokens": effective,
                "block_ids": ids, "bytes": resident * self.config.kv_bytes_per_token, "state_version": self.version}

    def storage_snapshot(self):
        blocks = [b for b in self.pool.blocks if not b.is_null]
        referenced = sum(b.ref_cnt > 0 for b in blocks)
        cached = sum(b.block_hash is not None and self.ready_hashes.get(b.block_id) == b.block_hash for b in blocks)
        evictable_cached = sum(b.ref_cnt == 0 and b.block_hash is not None and self.ready_hashes.get(b.block_id) == b.block_hash for b in blocks)
        size = self.config.kv_block_size_tokens * self.config.kv_bytes_per_token
        return {"source": "native_vllm_block_pool", "state_version": self.version,
                "capacity": self.scheduler.orbitserve_capacity, "total_blocks": self.pool.num_gpu_blocks,
                "usable_blocks": len(blocks), "referenced_unique_blocks": referenced,
                "referenced_bytes": referenced * size, "free_or_reclaimable_blocks": self.pool.get_num_free_blocks(),
                  "kv_cache_usage": self.scheduler.kv_cache_manager.usage,
                "ready_cached_blocks": cached, "evictable_cached_blocks": evictable_cached,
                "migration_reserved_blocks": sum(len(t["new"]) for t in self.imports.values()),
                "waiting_request_ids": [rid for rid, r in self.scheduler.requests.items() if r.status.name != "RUNNING"],
                "running_request_ids": [rid for rid, r in self.scheduler.requests.items() if r.status.name == "RUNNING"],
                "records": list(self.records)}

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
        payload = {"hashes": tuple(hashes), "block_bytes": self.config.kv_block_size_tokens * self.config.kv_bytes_per_token,
                   "history": history, "source_block_ids": tuple(b.block_id for b in blocks)}
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
        if missing and len(missing) + self.scheduler.kv_cache_manager.watermark_blocks > self.pool.get_num_free_blocks():
            self.pool.free_blocks(existing)
            return {"accepted": False, "reason": "target_capacity_temporarily_unavailable"}
        new = self.pool.get_new_blocks(len(missing))
        result = {"accepted": True, "missing_blocks": len(new), "bytes": len(new) * payload["block_bytes"]}
        self.imports[transfer_id] = {"existing": existing, "new": new, "hashes": missing, "result": result}
        self.version += 1
        return result

    def commit_import(self, transfer_id):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        state = self.imports.pop(transfer_id)
        for digest, block in zip(state["hashes"], state["new"], strict=True):
            # Version-specific adapter seam: attach imported symbolic data to native APC.
            self.pool._insert_block_hash(make_block_hash_with_group_id(digest, 0), block,
                                         self.config.kv_block_size_tokens)
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
        return {"removed_blocks": len(removable), "retained_blocks": len(state["blocks"]) - len(removable)}
