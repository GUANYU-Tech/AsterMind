import json

import pytest

from dataset_pipeline.serving_dataset.metax_c500_selected_step_mctracer import (
    _raw_eager_rank_trace,
    _raw_rank_trace,
)


def _graph_event(ts_ns, co_id, pid=17):
    return {
        "name": "mcGraphLaunch",
        "ph": "X",
        "ts": ts_ns,
        "dur": 10,
        "args": {
            "co_id": co_id,
            "process_id": pid,
            "ret": 0,
        },
    }


def _device_event(ts_ns, co_id, rank=0):
    return {
        "name": "kernel",
        "cat": "9" if rank == 1 else "0",
        "ph": "X",
        "ts": ts_ns,
        "dur": 50,
        "args": {"co_id": co_id, "device_id": rank, "grid": {}},
    }


def _host_launch_event(ts_ns, co_id, pid=17, duration_ns=10):
    return {
        "name": "mcLaunchKernel",
        "ph": "X",
        "ts": ts_ns,
        "dur": duration_ns,
        "args": {
            "co_id": co_id,
            "process_id": pid,
            "ret": 0,
            "link": True,
        },
    }


def _write_trace(path, events):
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


def _interposed(starts):
    return [
        {
            "launch_ordinal": ordinal,
            "host_monotonic_start_ns": start,
        }
        for ordinal, start in enumerate(starts)
    ]


def test_clock_mapping_uses_local_bracketing_launches_not_warmup_history(tmp_path):
    host_starts = [100_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000]
    offsets = [900_000_000, 1_000_000_000, 1_000_000_000, 1_000_000_000]
    raw_starts = [host + offset for host, offset in zip(host_starts, offsets)]
    events = [
        _device_event(raw_starts[2] + 100, 102, rank=1),
        _device_event(raw_starts[3] + 100, 103, rank=1),
    ]
    for ordinal, raw_start in enumerate(raw_starts):
        co_id = 100 + ordinal
        events.append(_graph_event(raw_start, co_id))
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, events)
    launches = _interposed(host_starts)

    result = _raw_rank_trace(
        trace_path,
        17,
        1,
        launches,
        {"vllm:10": launches[2], "vllm:11": launches[3]},
    )

    assert result["clock_mapping"] == {
        "method": "median_local_graph_launch_start_minus_interposer_start",
        "anchor_ordinals": [1, 2, 3],
        "raw_minus_host_monotonic_ns": 1_000_000_000,
        "maximum_residual_ns": 0,
        "p99_residual_ns": 0,
    }
    assert result["steps"]["vllm:10"]["intervals"] == [
        (3_000_000_100, 3_000_000_150)
    ]
    assert result["steps"]["vllm:10"]["task_groups"] == [{
        "task_type": "kernel",
        "name": "kernel",
        "launch": {
            "grid": {},
            "block": None,
            "dynamic_shared_bytes": None,
            "bytes": None,
            "copy_kind": None,
            "source_kind": None,
            "destination_kind": None,
        },
        "count": 1,
        "additive_duration_ns": 50,
    }]


def test_clock_mapping_rejects_local_anchor_disagreement(tmp_path):
    host_starts = [100_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000]
    offsets = [900_000_000, 1_000_000_000, 1_000_000_000, 1_003_000_000]
    raw_starts = [host + offset for host, offset in zip(host_starts, offsets)]
    events = []
    for ordinal, raw_start in enumerate(raw_starts):
        co_id = 100 + ordinal
        events.append(_graph_event(raw_start, co_id))
        if ordinal >= 2:
            events.append(_device_event(raw_start + 100, co_id))
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, events)
    launches = _interposed(host_starts)

    with pytest.raises(ValueError, match="local .* residual exceeds 1 ms"):
        _raw_rank_trace(
            trace_path,
            17,
            0,
            launches,
            {"vllm:10": launches[2], "vllm:11": launches[3]},
        )


def test_clock_mapping_rejects_first_graph_launch_without_preceding_anchor(
    tmp_path,
):
    host_starts = [2_000_000_000, 3_000_000_000]
    raw_starts = [host + 1_000_000_000 for host in host_starts]
    events = []
    for ordinal, raw_start in enumerate(raw_starts):
        co_id = 100 + ordinal
        events.append(_graph_event(raw_start, co_id))
        events.append(_device_event(raw_start + 100, co_id))
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, events)
    launches = _interposed(host_starts)

    with pytest.raises(ValueError, match="requires a preceding local clock anchor"):
        _raw_rank_trace(
            trace_path,
            17,
            0,
            launches,
            {"vllm:1": launches[0], "vllm:2": launches[1]},
            required_device_step_ids={"vllm:1"},
        )


def test_clock_mapping_rejects_nonadjacent_formal_launches(tmp_path):
    host_starts = [100_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000]
    raw_starts = [host + 1_000_000_000 for host in host_starts]
    events = [_graph_event(start, 100 + ordinal) for ordinal, start in enumerate(raw_starts)]
    events.extend(
        [
            _device_event(raw_starts[1] + 100, 101),
            _device_event(raw_starts[3] + 100, 103),
        ]
    )
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, events)
    launches = _interposed(host_starts)

    with pytest.raises(ValueError, match="must be adjacent"):
        _raw_rank_trace(
            trace_path,
            17,
            0,
            launches,
            {"vllm:10": launches[1], "vllm:11": launches[3]},
        )


def test_unanchored_graph_capture_exposes_per_rank_union_only(tmp_path):
    host_start = 2_000_000_000
    raw_start = host_start + 1_000_000_000
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, [
        _graph_event(raw_start, 100),
        _device_event(raw_start + 100, 100),
    ])
    launch = _interposed([host_start])[0]

    result = _raw_rank_trace(
        trace_path,
        17,
        0,
        [launch],
        {"vllm:1": launch},
        allow_unanchored_device_union=True,
    )

    assert result["steps"]["vllm:1"]["device_union_ns"] == 50
    assert result["clock_mapping"] == {
        "method": "single_graph_launch_start_alignment_unvalidated",
        "anchor_ordinals": [0],
        "raw_minus_host_monotonic_ns": 1_000_000_000,
        "maximum_residual_ns": None,
        "p99_residual_ns": None,
        "host_clock_comparable": False,
    }


def test_eager_step_uses_linked_host_apis_inside_projected_context(tmp_path):
    host_starts = [100_000_000, 200_000_000, 300_000_000]
    raw_starts = [host + 1_000_000_000 for host in host_starts]
    events = [
        _graph_event(raw_starts[0], 100),
        _host_launch_event(1_150_000_000, 201),
        _device_event(1_150_000_100, 201),
        _graph_event(raw_starts[1], 101),
        _device_event(raw_starts[1] + 100, 101),
        _graph_event(raw_starts[2], 102),
    ]
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, events)
    launches = _interposed(host_starts)

    result = _raw_rank_trace(
        trace_path,
        17,
        0,
        launches,
        {"vllm:11": launches[1]},
        required_device_step_ids={"vllm:10"},
        eager_contexts={
            "vllm:10": {
                "begin_ns": 140_000_000,
                "end_ns": 160_000_000,
            }
        },
    )

    assert result["steps"]["vllm:10"]["intervals"] == [
        (150_000_100, 150_000_150)
    ]
    assert result["clock_mapping"]["anchor_ordinals"] == [0, 1, 2]


def test_eager_trace_correlates_kernel_launch_ordinals(tmp_path):
    interposed = [
        {
            "launch_ordinal": ordinal,
            "api": "mcLaunchKernel",
            "host_monotonic_start_ns": 1000 + ordinal * 100,
            "grid": [4, 1, 1],
            "block": [64, 1, 1],
            "dynamic_shared_memory_bytes": 0,
        }
        for ordinal in range(3)
    ]
    events = []
    for ordinal in range(3):
        co_id = 20 + ordinal
        raw_start = 11_000 + ordinal * 100
        events.append({
            # An unrelated initialization launch may have a different API
            # ordering between mcTracer and the interposer.  The selected
            # ordinal range below must remain exact.
            "name": "mcModuleLaunchKernel" if ordinal == 0 else "mcLaunchKernel",
            "ph": "X",
            "ts": raw_start,
            "dur": 5,
            "args": {"co_id": co_id, "process_id": 17, "ret": 0},
        })
        if ordinal > 0:
            events.append({
                "name": "decoder_kernel",
                "ph": "X",
                "ts": raw_start + 10,
                "dur": 40 + ordinal,
                "args": {
                    "co_id": co_id,
                    "device_id": 0,
                    "grid": {"x": 4, "y": 1, "z": 1},
                    "block": {"x": 64, "y": 1, "z": 1},
                },
            })
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, events)

    result = _raw_eager_rank_trace(
        trace_path, 17, 0, interposed, interposed[1:]
    )

    assert result["target_kernel_count"] == 2
    assert result["device_kernel_union_ns"] == 83
    assert result["clock_mapping"]["maximum_residual_ns"] == 0
    assert result["kernel_groups"] == [{
        "name": "decoder_kernel",
        "launch": {
            "api": "mcLaunchKernel",
            "grid": [4, 1, 1],
            "block": [64, 1, 1],
            "dynamic_shared_bytes": 0,
        },
        "count": 2,
        "additive_duration_ns": 83,
        "duration_samples_ns": [41, 42],
    }]


def test_eager_trace_rejects_api_drift_inside_selected_context(tmp_path):
    interposed = [
        {
            "launch_ordinal": ordinal,
            "api": "mcLaunchKernel",
            "host_monotonic_start_ns": 1000 + ordinal * 100,
            "grid": [4, 1, 1],
            "block": [64, 1, 1],
            "dynamic_shared_memory_bytes": 0,
        }
        for ordinal in range(2)
    ]
    events = [
        {
            "name": "mcLaunchKernel" if ordinal == 0 else "mcModuleLaunchKernel",
            "ph": "X",
            "ts": 11_000 + ordinal * 100,
            "dur": 5,
            "args": {"co_id": 20 + ordinal, "process_id": 17, "ret": 0},
        }
        for ordinal in range(2)
    ]
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, events)

    with pytest.raises(ValueError, match="API identity differs"):
        _raw_eager_rank_trace(trace_path, 17, 0, interposed, [interposed[1]])
