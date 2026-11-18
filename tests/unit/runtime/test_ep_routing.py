"""Direct DeepSeek V4 EP uses the same deterministic routing as expert compute."""

from types import SimpleNamespace

from astermind.builder.config.communication import CommPolicy


def test_direct_dispatch_deduplicates_experts_on_the_same_rank():
    from astermind.common.bootstrap import activate

    activate()
    from astermind.engine.compute.ep_routing import _token_matrix

    routes = ((0, 2, 3, 6), (4, 5), (2, 3, 7))
    assert _token_matrix(routes, (2, 1, 0, 0), 2) == [
        [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
    ]


def test_v4_plan_dispatch_and_combine_use_transposed_matrices():
    from astermind.common.bootstrap import activate

    root = activate()
    from astermind.engine.compute.ep_routing import direct_ep_matrices
    from astermind.engine.compute.runtime import (
        MidlayerCellConfig, create_midlayer_runtime, prepare_scheduled_step,
    )
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest

    assert CommPolicy().ep_strategy == "allgather"
    config = MidlayerCellConfig(
        root, model_file="configs/model/deepseek-v4-flash.json",
        hardware_file="configs/hardware/nvidia_h200.yaml", hardware_id="nvidia_h200",
        scheduler_adapter="vllm_0_30", tensor_parallel_size=2, data_parallel_size=2,
        enable_expert_parallel=True, expert_routing_seed=42,
        max_model_len=256, max_batched_tokens=16, kv_block_size_tokens=256,
    )
    _, execution, _ = create_midlayer_runtime(config, with_scheduler=False)
    schedule = ScheduleStep("ep-test", "main", tuple(
        ScheduledRequest(f"r{i}", "prefill", 2, 0, data_parallel_rank=i) for i in range(2)
    ))
    step = SimpleNamespace(schedule=schedule, embedding_token_ids=(42,) * schedule.token_count,
                           model_source_inputs=None, kv_cache_events=())
    plan = prepare_scheduled_step(execution, step, (2, 2))
    calls = {call.call_id: call for call in plan.calls}
    matrices = direct_ep_matrices(calls, execution, "main", schedule.step_id)

    assert len(matrices) == 4 * execution.model.num_hidden_layers
    for name, dispatch in matrices.items():
        if not name.endswith(":dispatch-hidden"):
            continue
        combine = matrices[name.removesuffix("dispatch-hidden") + "combine"]
        assert combine == tuple(zip(*dispatch))
        assert len(dispatch) == 4
        assert all(row[i] == 0 for i, row in enumerate(dispatch))
        assert any(any(row) for row in dispatch)
        assert all(max(row) <= size for row, size in zip(
            dispatch, calls[name].kernel_spec.transfer.partition_bytes, strict=True))
