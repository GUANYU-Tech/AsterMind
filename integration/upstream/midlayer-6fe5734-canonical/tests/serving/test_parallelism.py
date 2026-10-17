import pytest

from src.serving.parallelism.parallel_config import ParallelConfig, ParallelConfigError
from src.serving.parallelism.parallel_plan import build_parallel_plan


def test_rank_groups_reuse_tensor_ranks_for_sequence_parallelism():
    config = ParallelConfig(
        tensor_parallel_size=4, sequence_parallel_size=2,
        pipeline_parallel_size=2, data_parallel_size=2)
    plan = build_parallel_plan(config, num_layers=8)
    assert plan.rank_groups.world_size == 16
    assert all(len(group) == 4 for group in plan.rank_groups.tensor)
    assert all(len(group) == 2 for group in plan.rank_groups.sequence)
    assert plan.layer_to_pipeline_stage == (0, 0, 0, 0, 1, 1, 1, 1)


def test_expert_parallelism_reuses_tp_ranks_with_linear_placement():
    config = ParallelConfig(
        tensor_parallel_size=2,
        enable_expert_parallel=True,
    )
    plan = build_parallel_plan(config, num_layers=2, num_experts=6)

    assert config.worker_count == 2
    assert config.expert_parallel_size == 2
    assert plan.rank_groups.tensor == ((0, 1),)
    assert plan.rank_groups.expert == ((0, 1),)
    assert plan.expert_to_shard == (0, 0, 0, 1, 1, 1)
    assert plan.communication_rules == (
        "row_parallel_all_reduce",
        "vocab_parallel_all_gather",
        "expert_output_all_reduce",
    )


def test_expert_parallelism_rejects_uneven_expert_placement():
    config = ParallelConfig(
        tensor_parallel_size=2,
        enable_expert_parallel=True,
    )

    with pytest.raises(ValueError, match="divisible"):
        build_parallel_plan(config, num_layers=2, num_experts=5)


def test_expert_parallelism_reuses_dp_and_tp_ranks_per_pipeline_stage():
    config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
    )
    plan = build_parallel_plan(config, num_layers=2, num_experts=4)

    assert config.worker_count == 8
    assert config.expert_parallel_size == 4
    assert plan.rank_groups.expert == ((0, 1, 4, 5), (2, 3, 6, 7))
    assert plan.communication_rules == (
        "row_parallel_all_reduce",
        "vocab_parallel_all_gather",
        "pipeline_activation_send",
        "expert_dispatch_all_gatherv",
        "expert_combine_reduce_scatterv",
        "expert_output_all_reduce",
    )


def test_dp_only_expert_parallelism_uses_default_dispatch_and_combine():
    config = ParallelConfig(
        data_parallel_size=2,
        enable_expert_parallel=True,
    )
    plan = build_parallel_plan(config, num_layers=2, num_experts=2)

    assert config.worker_count == 2
    assert plan.rank_groups.expert == ((0, 1),)
    assert plan.communication_rules == (
        "expert_dispatch_all_gatherv",
        "expert_combine_reduce_scatterv",
    )


def test_disabled_expert_parallel_groups_are_effective_singletons():
    plan = build_parallel_plan(
        ParallelConfig(tensor_parallel_size=2),
        num_layers=2,
        num_experts=2,
    )

    assert plan.rank_groups.expert == ((0,), (1,))
    assert plan.expert_to_shard == (0, 0)


def test_expert_parallelism_requires_more_than_one_reused_rank():
    with pytest.raises(
            ParallelConfigError,
            match=r"requires TP \* DP greater than one"):
        ParallelConfig(enable_expert_parallel=True)


@pytest.mark.parametrize(
    "kwargs,error",
    (
        ({"expert_communication_backend": "pplx"},
         "only the allgather_reducescatter backend"),
        ({"enable_eplb": True}, "does not yet support EPLB"),
    ),
)
def test_unvalidated_expert_runtime_modes_fail_loud(kwargs, error):
    with pytest.raises(ParallelConfigError, match=error):
        ParallelConfig(**kwargs)


def test_pipeline_stage_lookup_returns_all_logical_ranks_at_that_stage():
    plan = build_parallel_plan(
        ParallelConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            data_parallel_size=2,
        ),
        num_layers=8,
    )

    assert plan.logical_ranks_for_pipeline_stage(0) == (
        0, 1, 2, 3, 8, 9, 10, 11,
    )
    assert plan.logical_ranks_for_pipeline_stage(1) == (
        4, 5, 6, 7, 12, 13, 14, 15,
    )


@pytest.mark.parametrize("stage_index", (True, 0.0, "0"))
def test_pipeline_stage_lookup_rejects_non_integer_index(stage_index):
    plan = build_parallel_plan(
        ParallelConfig(pipeline_parallel_size=2), num_layers=2)

    with pytest.raises(TypeError, match="stage index must be an integer"):
        plan.logical_ranks_for_pipeline_stage(stage_index)


@pytest.mark.parametrize("stage_index", (-1, 2))
def test_pipeline_stage_lookup_rejects_out_of_range_index(stage_index):
    plan = build_parallel_plan(
        ParallelConfig(pipeline_parallel_size=2), num_layers=2)

    with pytest.raises(ValueError, match="outside the configured pipeline"):
        plan.logical_ranks_for_pipeline_stage(stage_index)


def test_unsupported_context_parallel_fails_when_requested():
    with pytest.raises(NotImplementedError, match="Context Parallel"):
        build_parallel_plan(
            ParallelConfig(context_parallel_size=2), num_layers=8)


def test_invalid_sequence_parallel_reuse_fails_during_config_validation():
    with pytest.raises(ParallelConfigError, match="divisible"):
        ParallelConfig(tensor_parallel_size=4, sequence_parallel_size=3)


@pytest.mark.parametrize("dimension", ("world_size", "tensor_core", "unknown"))
def test_rank_group_lookup_rejects_unknown_dimension(dimension):
    groups = build_parallel_plan(
        ParallelConfig(tensor_parallel_size=2), num_layers=1,
    ).rank_groups
    with pytest.raises(ValueError, match="unknown parallel dimension"):
        groups.for_rank(dimension, 0)
