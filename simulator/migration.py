"""Native block transaction with actual shared-network delivery before commit."""


class NetworkKVMigration:
    def __init__(self, transport, endpoints):
        self.loop, self.transport, self.endpoints = transport.loop, transport, endpoints
        self.records = {}
        self._waiting_at_boundary = []

    def on_batch_boundary(self):
        for begin_if_idle in tuple(self._waiting_at_boundary):
            begin_if_idle()

    def start(self, spec, source, target, history):
        from orbitserve.runtime.engine.cell_events import STEP_COMPLETE
        if spec.id in self.records:
            raise ValueError("migration already started")
        if source.storage_signature != target.storage_signature:
            raise ValueError("migration requires matching model, KV format and TP/PP partitions")
        def ranks(cell):
            value = self.endpoints[cell.cell_id]
            return {0: value} if isinstance(value, str) else dict(value)
        source_devices, target_devices = ranks(source), ranks(target)
        if not source_devices or source_devices.keys() != target_devices.keys():
            raise ValueError("migration needs matching physical rank endpoint maps")
        # Validate every structural route before taking native block references.
        for rank in source_devices:
            self.transport.network.candidate_path(source_devices[rank], target_devices[rank])
        record = dict(id=spec.id, source=source.cell_id, target=target.cell_id,
                      state="waiting_for_batch_boundary", requested_time_s=self.loop.now_s)
        self.records[spec.id] = record
        source.pause_for_migration(spec.id)
        try:
            target.pause_for_migration(spec.id)
        except BaseException:
            source.resume_after_migration(spec.id)
            raise
        def release():
            source.resume_after_migration(spec.id)
            target.resume_after_migration(spec.id)
        def begin_if_idle():
            if source._inflight is not None or target._inflight is not None:
                return
            self._waiting_at_boundary.remove(begin_if_idle)
            exported = False
            reserved = False
            try:
                payload = source.storage_rpc("export_prefix", spec.id, history, spec.mode)
                exported = True
                imported = target.storage_rpc("reserve_import", spec.id, payload)
                if not imported["accepted"]:
                    source.storage_rpc("release_export", spec.id, False)
                    record.update(state="capacity_blocked", reason=imported["reason"])
                    release()
                    return
                reserved = True
                size = imported["bytes"]
                partitions = [dict(rank=rank, source=source_devices[rank], target=target_devices[rank],
                    bytes=size, transfer_id=f"migration-{spec.id}" if len(source_devices) == 1
                    else f"migration-{spec.id}-rank-{rank}") for rank in sorted(source_devices)]
                record.update(state="transferring", bytes=size * len(partitions), bytes_per_rank=size,
                              partitions=partitions, start_time_s=self.loop.now_s)
                pending = set(source_devices)
                failures = []
                def delivered(rank, failure=None):
                    if rank not in pending:
                        raise RuntimeError("duplicate KV partition terminal event")
                    pending.remove(rank)
                    part = next(p for p in partitions if p["rank"] == rank)
                    part.update(state="failed" if failure else "completed", terminal_time_s=self.loop.now_s)
                    if failure:
                        failures.append(failure)
                    if pending:
                        return
                    if failures:
                        target.storage_rpc("rollback_import", spec.id)
                        source.storage_rpc("release_export", spec.id, False)
                        record.update(state="failed", failures=failures, completed_time_s=self.loop.now_s)
                        release()
                        return
                    record["network_complete_time_s"] = self.loop.now_s
                    def commit():
                        target.storage_rpc("commit_import", spec.id)
                        result = source.storage_rpc("release_export", spec.id, spec.mode == "move")
                        record.update(state="completed", source_release=result, completed_time_s=self.loop.now_s)
                        release()
                    self.loop.schedule(self.loop.now_s + target.migration_memory_time(size, "write"),
                                       STEP_COMPLETE, f"migration:{spec.id}:commit", commit)
                def send():
                    for partition in partitions:
                        rank = partition["rank"]
                        try:
                            self.transport.submit_transport(partition["transfer_id"], partition["source"],
                                partition["target"], size, "kv_migration", spec.id,
                                lambda rank=rank: delivered(rank),
                                lambda event, rank=rank: delivered(rank, event.model_dump()))
                        except Exception as error:
                            # Other submitted partitions still own source data.
                            # Roll back only after every partition is terminal.
                            delivered(rank, dict(reason="submission_failed", error=str(error)))
                self.loop.schedule(self.loop.now_s + source.migration_memory_time(size, "read"),
                                   STEP_COMPLETE, f"migration:{spec.id}:send", send)
            except BaseException:
                if reserved:
                    target.storage_rpc("rollback_import", spec.id)
                if exported:
                    source.storage_rpc("release_export", spec.id, False)
                record.update(state="failed")
                release()
                raise
        self._waiting_at_boundary.append(begin_if_idle)
        begin_if_idle()
        return record
