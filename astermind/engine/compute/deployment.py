"""OrbitServe Cell checks; does not change ArchPrism model construction."""
def validate_cell_deployment(config, execution, placements):
    mapping={(p.pool_name,p.rank):p.gpu_device_id for p in placements}
    for name,pool in execution.deployment.worker_pools.items():
        if any((name,r) not in mapping for r in pool.ranks):
            raise ValueError("actual worker pool contains an unmapped rank")
        parallel=pool.parallel_plan.config
        if parallel.data_parallel_size!=config.parallel.dp:
            raise ValueError("actual DP groups differ from Cell configuration")
        if not config.wide_ep_enabled:continue
        for stage in range(parallel.pipeline_parallel_size):
            members=set(pool.physical_ranks_for_pipeline_stage(stage))
            tensor_groups={tuple(pool.physical_rank_group("tensor",pool.logical_rank_for_physical_rank(r))) for r in members}
            if len(tensor_groups)!=parallel.data_parallel_size:
                raise ValueError("WideEP stage must cover every DP replica")
            for rank in members:
                group=set(pool.physical_rank_group("expert",pool.logical_rank_for_physical_rank(rank)))
                if group!=members or len(group)!=config.parallel.ep:
                    raise ValueError("actual WideEP group differs from stage-local declaration")
