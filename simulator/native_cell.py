"""External storage layout description; native computation is unchanged."""
from midlayer_cell import MidlayerCell


class NetworkMidlayerCell(MidlayerCell):
    @property
    def storage_signature(self):
        """Require identical logical block and per-rank TP/PP partitions.

        One native allocator manages logical blocks across all worker ranks.
        NetworkKVMigration transfers each physical rank's bytes before commit.
        This does not implement redistribution between different partitions.
        """
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
