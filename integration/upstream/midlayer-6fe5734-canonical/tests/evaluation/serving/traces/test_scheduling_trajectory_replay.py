"""Fail-closed contracts for fixed Scheduler trajectory replay."""

from copy import deepcopy

import pytest

from evaluation.serving.traces import scheduling_trajectory_replay as replay
from src.serving.workload.graph_padding import GraphExecutionConfig


def _fingerprint():
    return {
        "schema_version": 1,
        "cell_id": "metax_c500/qwen3-30b-a3b/tp2_ep2_qps4",
        "hardware_id": "metax_c500",
        "model_id": "qwen3-30b-a3b",
        "step_id": "vllm:1085",
        "request_workload": [{
            "request_idx": 4,
            "request_id": "request-000000000004",
            "computed_tokens_before_step": 653,
            "kv_context_tokens_after_step": 654,
            "prefill_tokens": 0,
            "decode_tokens": 1,
            "scheduled_tokens": 1,
        }],
        "token_totals": {
            "prefill_tokens": 0,
            "decode_tokens": 1,
            "scheduled_tokens": 1,
        },
        "parallelism": {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 2,
        },
        "graph_execution": {
            "configured_mode": "FULL_AND_PIECEWISE",
            "runtime_mode": "FULL",
            "uniform_decode": True,
            "logical_tokens": 1,
            "model_input_tokens": 1,
            "capture_bucket_tokens": 1,
            "capture_limit_tokens": 512,
            "padding_tokens": 0,
            "fallback_reason": None,
            "rank_ids": [0, 1],
        },
        "normalized_launch_program_sha256": "a" * 64,
        "normalized_launch_call_count": 117,
    }


@pytest.mark.parametrize(
    ("path", "value", "field"),
    (
        (("graph_execution", "capture_bucket_tokens"), 4,
         "fingerprint.graph_execution.capture_bucket_tokens"),
        (("graph_execution", "padding_tokens"), 3,
         "fingerprint.graph_execution.padding_tokens"),
        (("request_workload", 0, "computed_tokens_before_step"), 654,
         "fingerprint.request_workload[0].computed_tokens_before_step"),
    ),
)
def test_step_fingerprint_mismatch_is_invalid_experiment(path, value, field):
    observed = _fingerprint()
    replayed = deepcopy(observed)
    target = replayed
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValueError) as error:
        replay.require_exact_step_fingerprints(observed, replayed)
    assert f"INVALID EXPERIMENT — {field}" in str(error.value)


def test_step_fingerprint_exact_match_is_accepted():
    fingerprint = _fingerprint()
    replay.require_exact_step_fingerprints(fingerprint, deepcopy(fingerprint))


def test_graph_on_observation_survives_replay_identity_validation():
    row = {
        "requests": [{"scheduled_tokens": 1} for _ in range(33)],
        "scheduled_total_tokens": 33,
        "graph_execution_observation": {
            "configured_mode": "FULL_AND_PIECEWISE",
            "runtime_mode": "FULL",
            "uniform_decode": True,
            "logical_tokens": 33,
            "model_input_tokens": 40,
            "capture_bucket_tokens": 40,
            "capture_limit_tokens": 512,
            "padding_tokens": 7,
            "fallback_reason": None,
            "rank_ids": [],
        },
    }
    config = GraphExecutionConfig(
        "FULL_AND_PIECEWISE",
        (1, 40, 512),
        ("flash_attn", "kv_update"),
    )

    identity = replay._validated_replay_graph_identity(
        row, config, expected_rank_ids=(0, 1)
    )

    assert identity["configured_mode"] == "FULL_AND_PIECEWISE"
    assert identity["runtime_mode"] == "FULL"
    assert identity["capture_bucket_tokens"] == 40
    assert identity["padding_tokens"] == 7
    assert identity["rank_ids"] == [0, 1]


def test_missing_graph_observation_is_invalid_experiment():
    row = {
        "requests": [{"scheduled_tokens": 1}],
        "scheduled_total_tokens": 1,
        "graph_execution_observation": None,
    }

    with pytest.raises(ValueError) as error:
        replay._validated_replay_graph_identity(
            row,
            GraphExecutionConfig("FULL", (1,), ()),
            expected_rank_ids=(0,),
        )
    assert "INVALID EXPERIMENT — graph_execution_observation" in str(
        error.value
    )


def test_report_gate_rejects_incomplete_fingerprint():
    observed = _fingerprint()
    replayed = deepcopy(observed)
    del replayed["normalized_launch_program_sha256"]

    with pytest.raises(ValueError) as error:
        replay.require_exact_step_fingerprints(observed, replayed)
    assert "INVALID EXPERIMENT — fingerprint.fields" in str(error.value)


def test_graph_replay_barrier_diagnostic_carries_cross_step_device_completion():
    rows = []
    for index, runtime_mode in enumerate(("NONE", "FULL", "FULL")):
        rows.append({
            "step_index": index,
            "step_id": f"vllm:{index + 1}",
            "step_start_ns": index * 10,
            "schedule_end_ns": index * 10 + 1,
            "update_start_ns": index * 10 + 2,
            "step_end_ns": index * 10 + 3,
            "requests": ({
                "request_idx": 0,
                "prefill_tokens": 0,
                "decode_tokens": 1,
                "scheduled_tokens": 1,
            },),
            "graph_execution_observation": {
                "runtime_mode": runtime_mode,
            },
        })

    _completion, timing = replay.replay_scheduling_trajectory(
        rows,
        [{"arrival_time_s": 0.0}],
        [1.0, 1.0, 1.0],
        include_observed_scheduler_cpu=False,
        pipeline_depth=2,
        include_graph_replay_barrier=True,
    )

    assert timing["graph_replay_barrier_enabled"] is True
    assert timing["graph_replay_barrier_wait_count"] == 2
    assert timing["graph_replay_barrier_wait_total_s"] == 2.0
    assert timing["graph_replay_barrier_waits_s"] == [0.0, 1.0, 1.0]
