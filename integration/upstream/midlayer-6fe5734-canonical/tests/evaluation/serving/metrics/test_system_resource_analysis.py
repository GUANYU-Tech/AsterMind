from types import SimpleNamespace

from evaluation.serving.metrics.system_resource_analysis import (
    _request_execution_blocks, _request_target_path,
)
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduleStep,
    ScheduledRequest,
)


def test_request_critical_path_excludes_discarded_async_terminal_tail():
    completed_request = ScheduledRequest(
        "request", "decode", 0, 1, 128, raw_model_output_tokens=1)
    completed_step = BackendStep(
        ScheduleStep(
            "pool:completed",
            "pool",
            (completed_request,),
            finished_request_ids=("request",),
        ),
        generated_tokens={"request": 1},
    )
    discarded_request = ScheduledRequest(
        "request", "decode", 0, 1, 129, raw_model_output_tokens=1)
    discarded_step = BackendStep(
        ScheduleStep("pool:discarded", "pool", (discarded_request,)),
        generated_tokens={"request": 0},
    )
    entries = {
        "pool:completed": SimpleNamespace(
            backend_step=completed_step, start_time_s=0.0, end_time_s=2.0),
        "pool:discarded": SimpleNamespace(
            backend_step=discarded_step, start_time_s=0.5, end_time_s=1.5),
    }
    paths = {
        step_id: {"segments": ()}
        for step_id in entries
    }
    result = SimpleNamespace(
        schedule_trace=SimpleNamespace(route_decisions=(), transfers=()),
        transfers=(),
    )

    blocks = _request_execution_blocks(
        "request",
        SimpleNamespace(deployment_id="test", routes=()),
        result,
        entries,
        paths,
        2.0,
    )

    assert [block["step_id"] for block in blocks] == ["pool:completed"]


def test_ttft_path_represents_host_delivery_overlapping_next_batch():
    blocks = [
        {
            "type": "step",
            "step_id": "decode:first",
            "start_time_s": 0.0,
            "end_time_s": 2.0,
            "segments": ({
                "type": "kernel",
                "category": "kernel:compute",
                "duration_s": 2.0,
            },),
        },
        {
            "type": "step",
            "step_id": "decode:next",
            "start_time_s": 2.1,
            "end_time_s": 3.0,
            "segments": (),
        },
    ]

    path = _request_target_path("request", 0.0, 2.25, blocks, "ttft")

    assert [segment["type"] for segment in path["segments"]] == [
        "shared_batch_step", "host_first_token_delivery",
    ]
    assert path["segments"][-1]["duration_s"] == 0.25
    assert path["contribution_s_by_category"] == {
        "host:first_token_delivery": 0.25,
        "kernel:compute": 2.0,
    }
