"""Dispatch Fast Mode session histories to the native Scheduler that owns them."""

from src.serving.backend.vllm import _DataParallelControlPlane


class SessionDataParallel(_DataParallelControlPlane):
    def __init__(self, replicas, *, pool_name, source_scheduler_config):
        super().__init__(replicas, pool_name=pool_name, source_scheduler_config=source_scheduler_config)
        self._session_owners = {}

    def rpc(self, name, *args):
        if name == "prepare_history":
            request_id, history = args
            if request_id in self._owners:
                raise ValueError("duplicate request history")
            session_id = history["session_id"]
            if session_id in self._session_owners:
                owner = self._session_owners[session_id]
            else:
                owner = self._next_replica
            self.replicas[owner]._control_plane.rpc(name, request_id, history)
            if session_id not in self._session_owners:
                self._session_owners[session_id] = owner
                self._next_replica = (owner + 1) % len(self.replicas)
            self._owners[request_id] = owner
            return None
        if name == "inspect_prefix":
            owner = self._session_owners.get(args[0]["session_id"])
            if owner is None:
                return {"resident_prefix_tokens": 0, "effective_prefix_hit_tokens": 0,
                        "block_ids": [], "bytes": 0, "state_version": 0}
            return self.replicas[owner]._control_plane.rpc(name, *args)
        if name == "abort_request":
            owner = self.request_data_parallel_rank(args[0])
            return self.replicas[owner]._control_plane.rpc(name, *args)
        if name == "storage_snapshot":
            replicas = [replica._control_plane.rpc(name, *args) for replica in self.replicas]
            state = {
                "referenced_bytes": sum(replica["referenced_bytes"] for replica in replicas),
                "free_or_reclaimable_blocks": sum(replica["free_or_reclaimable_blocks"] for replica in replicas),
                "waiting_request_ids": [rid for replica in replicas for rid in replica["waiting_request_ids"]],
            }
            if not args or not args[0]:
                state["replicas"] = replicas
                state["records"] = [record for replica in replicas for record in replica["records"]]
            return state
        raise ValueError(f"unsupported DP session storage command {name!r}")
