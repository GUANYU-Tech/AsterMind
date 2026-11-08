"""Native block transaction, adapted from Merge_ML_Version's migration seam."""

import math


class NetworkKVMigration:
    def __init__(self, loop, executor, endpoints):
        self.loop, self.executor, self.endpoints = loop, executor, endpoints
        self.records = {}
        self._cleanup = {}

    def close(self):
        for cleanup in tuple(self._cleanup.values()):
            cleanup("migration_runtime_closed")

    def start(self, spec, source, target, history, *, on_complete=None):
        from .cell_events import STEP_COMPLETE

        if spec.mode not in {"copy", "move"} or source is target:
            raise ValueError("migration requires distinct Cells and copy/move mode")
        if spec.id in self.records:
            raise ValueError("migration already started")
        if source.storage_signature != target.storage_signature:
            raise ValueError(
                "migration requires matching model, KV format and TP/PP partitions"
            )

        def ranks(cell):
            value = self.endpoints[cell.cell_id]
            return {0: value} if isinstance(value, str) else dict(value)

        source_devices, target_devices = ranks(source), ranks(target)
        if not source_devices or source_devices.keys() != target_devices.keys():
            raise ValueError("migration needs matching physical rank endpoint maps")
        # Validate every structural route before taking native block references.
        for rank in source_devices:
            self.executor.network.candidate_path(
                source_devices[rank], target_devices[rank]
            )
        record = {
            "id": spec.id,
            "source": source.cell_id,
            "target": target.cell_id,
            "state": "waiting_for_batch_boundary",
            "requested_time_s": self.loop.now_s,
        }
        self.records[spec.id] = record
        source.pause_for_migration(spec.id)
        try:
            target.pause_for_migration(spec.id)
        except BaseException:
            source.resume_after_migration(spec.id)
            raise

        def release():
            self._cleanup.pop(spec.id, None)
            if begin_if_idle in self.executor.boundary_listeners:
                self.executor.boundary_listeners.remove(begin_if_idle)
            try:
                source.resume_after_migration(spec.id)
            finally:
                target.resume_after_migration(spec.id)

        exported = False
        reserved = False
        pending = set()
        partitions = []

        def abort(reason):
            nonlocal exported, reserved
            if record["state"] in {"completed", "failed", "capacity_blocked"}:
                return
            # Native references may only be released after every shard is terminal.
            if pending:
                self.executor.network.cancel(
                    [p["transfer_id"] for p in partitions], reason
                )
                if pending:
                    return
                if record["state"] == "failed":
                    return
            try:
                if reserved:
                    target.storage_rpc("rollback_import", spec.id)
                    reserved = False
                if exported:
                    source.storage_rpc("release_export", spec.id, False)
                    exported = False
            finally:
                record.update(
                    state="failed", reason=reason, completed_time_s=self.loop.now_s
                )
                release()

        def memory_delay(cell, size, operation):
            value = cell.migration_memory_time(size, operation)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "migration memory delay must be finite and nonnegative"
                )
            return value

        def begin_if_idle():
            nonlocal exported, reserved, pending, partitions
            if source._inflight is not None or target._inflight is not None:
                return
            self.executor.boundary_listeners.remove(begin_if_idle)
            try:
                payload = source.storage_rpc(
                    "export_prefix", spec.id, history, spec.mode
                )
                exported = True
                imported = target.storage_rpc("reserve_import", spec.id, payload)
                if not imported["accepted"]:
                    source.storage_rpc("release_export", spec.id, False)
                    record.update(state="capacity_blocked", reason=imported["reason"])
                    release()
                    return
                reserved = True
                size = imported["bytes"]
                if type(size) is not int or size < 0:
                    raise ValueError(
                        "native migration bytes must be a nonnegative integer per rank"
                    )
                partitions = [
                    {
                        "rank": rank,
                        "source": source_devices[rank],
                        "target": target_devices[rank],
                        "bytes": size,
                        "transfer_id": f"migration-{spec.id}"
                        if len(source_devices) == 1
                        else f"migration-{spec.id}-rank-{rank}",
                    }
                    for rank in sorted(source_devices)
                ]
                record.update(
                    state="transferring",
                    bytes=size * len(partitions),
                    bytes_per_rank=size,
                    partitions=partitions,
                    start_time_s=self.loop.now_s,
                )
                failures = []

                def delivered(rank, failure=None):
                    if rank not in pending:
                        raise RuntimeError("duplicate KV partition terminal event")
                    pending.remove(rank)
                    part = next(p for p in partitions if p["rank"] == rank)
                    part.update(
                        state="failed" if failure else "completed",
                        terminal_time_s=self.loop.now_s,
                    )
                    if failure:
                        failures.append(failure)
                    if pending:
                        return
                    if failures:
                        record["failures"] = failures
                        abort("partition_transfer_failed")
                        return
                    record["network_complete_time_s"] = self.loop.now_s
                    record["state"] = "writing"

                    def commit():
                        if record["state"] != "writing":
                            return
                        try:
                            target.storage_rpc("commit_import", spec.id)
                        except BaseException:
                            abort("target_commit_failed")
                            raise
                        try:
                            result = source.storage_rpc(
                                "release_export", spec.id, spec.mode == "move"
                            )
                            record.update(
                                state="completed",
                                source_release=result,
                                completed_time_s=self.loop.now_s,
                            )
                        except BaseException:
                            record.update(
                                state="source_release_failed",
                                completed_time_s=self.loop.now_s,
                            )
                            raise
                        finally:
                            release()
                        if on_complete is not None:
                            on_complete(record)

                    self.loop.schedule(
                        self.loop.now_s + memory_delay(target, size, "write"),
                        STEP_COMPLETE,
                        f"migration:{spec.id}:commit",
                        commit,
                    )

                def send():
                    nonlocal pending
                    if record["state"] != "transferring":
                        return
                    pending = set(source_devices)
                    for partition in partitions:
                        rank = partition["rank"]
                        try:
                            self.executor.submit_transport(
                                partition["transfer_id"],
                                partition["source"],
                                partition["target"],
                                size,
                                "kv_migration",
                                spec.id,
                                lambda rank=rank: delivered(rank),
                                lambda event, rank=rank: delivered(
                                    rank, event.model_dump()
                                ),
                            )
                        except Exception as error:  # noqa: BLE001 -- wait for all shard failures before rollback
                            # Other submitted partitions still own source data.
                            # Roll back only after every partition is terminal.
                            delivered(
                                rank,
                                {"reason": "submission_failed", "error": str(error)},
                            )

                self.loop.schedule(
                    self.loop.now_s + memory_delay(source, size, "read"),
                    STEP_COMPLETE,
                    f"migration:{spec.id}:send",
                    send,
                )
            except BaseException:
                abort("migration_setup_failed")
                raise

        self._cleanup[spec.id] = abort
        self.executor.boundary_listeners.append(begin_if_idle)
        begin_if_idle()
        return record
