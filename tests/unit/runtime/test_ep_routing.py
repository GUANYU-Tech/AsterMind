"""Runtime WideEP transfers use the same deterministic routing as expert compute."""

from types import SimpleNamespace

import pytest

from astermind.builder.config.communication import CommPolicy


@pytest.mark.parametrize("legacy", ("auto", "allgather"))
def test_wideep_policy_normalizes_legacy_strategies(legacy):
    assert CommPolicy().ep_strategy == "direct"
    assert CommPolicy(ep_strategy=legacy).ep_strategy == "direct"


def test_direct_dispatch_deduplicates_experts_on_the_same_rank():
    from astermind.common.bootstrap import activate

    activate()
    from astermind.engine.compute.ep_routing import _token_matrix

    routes = iter(((0, 2, 3, 6), (4, 5), (2, 3, 7), (0, 6)))
    matrix, counts = _token_matrix(routes, (2, 2, 0, 0), (0, 0, 1, 1, 2, 2, 3, 3), (2, 1, 0, 0))
    assert matrix == [
        [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
    ]
    assert counts == (2, 0, 2, 2, 1, 1, 2, 1)


@pytest.mark.parametrize(('model', 'mode'), (
    ('deepseek-v4-flash', 'accelerated'), ('glm-5.2', 'accelerated'),
    ('glm-5.2', 'fast'), ('glm-5.3', 'accelerated'), ('qwen3-30b-a3b', 'accelerated'),
    ('kimi-k2.6', 'accelerated'),
))
def test_wideep_plan_dispatch_and_combine_use_transposed_matrices(model, mode):
    from astermind.common.bootstrap import activate

    root = activate()
    from astermind.engine.compute.ep_routing import direct_ep_matrices
    from astermind.engine.compute.runtime import (
        MidlayerCellConfig, create_midlayer_runtime, prepare_scheduled_step,
    )
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest

    config = MidlayerCellConfig(
        root, model_file=f"configs/model/{model}.json",
        hardware_file="configs/hardware/nvidia_h200.yaml", hardware_id="nvidia_h200",
        scheduler_adapter="vllm_0_30", tensor_parallel_size=2, data_parallel_size=2,
        enable_expert_parallel=True, expert_routing_seed=42, execution_mode=mode,
        max_model_len=256, max_batched_tokens=16, kv_block_size_tokens=256,
    )
    _, execution, _ = create_midlayer_runtime(config, with_scheduler=False)
    schedule = ScheduleStep("ep-test", "main", tuple(
        ScheduledRequest(f"r{i}", "prefill", count, 0, data_parallel_rank=i) for i, count in enumerate((3, 1))
    ))
    step = SimpleNamespace(schedule=schedule, embedding_token_ids=(42,) * schedule.token_count,
                           model_source_inputs=None, kv_cache_events=())
    plan = prepare_scheduled_step(execution, step, (3, 1))
    calls = {call.call_id: call for call in plan.calls}
    matrices = direct_ep_matrices(calls, execution, "main", schedule.step_id, replica_tokens=(3, 1),
                                  source_nodes=plan.wide_ep_source_nodes)

    combines = [call for name, call in calls.items() if name in matrices
                and call.kernel_spec.transfer.operation == 'reduce_scatterv']
    assert len(combines) == sum(execution.model.layer_uses_moe(layer)
                                for layer in range(execution.model.num_hidden_layers))
    assert len(matrices) == (3 if model.startswith('qwen') else 4) * len(combines)
    for name, dispatch in matrices.items():
        if not name.endswith(("dispatch-hidden", "dispatch-input")):
            continue
        combine = matrices[name.rsplit("dispatch-", 1)[0] + "combine"]
        assert combine == tuple(zip(*dispatch))
        assert len(dispatch) == 4
        assert all(row[i] == 0 for i, row in enumerate(dispatch))
        assert any(any(row) for row in dispatch)
        assert not any(dispatch[3])
        assert all(max(row) <= size for row, size in zip(
            dispatch, calls[name].kernel_spec.transfer.partition_bytes, strict=True))
