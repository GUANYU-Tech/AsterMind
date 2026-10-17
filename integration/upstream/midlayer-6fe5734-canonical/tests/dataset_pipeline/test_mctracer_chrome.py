import json
import time

import pytest

import dataset_pipeline.conversion.mctracer_to_nsys as mctracer_to_nsys
from dataset_pipeline.conversion.mctracer_chrome import (
    McTracerKernelEvent,
    McTracerMemcpyEvent,
    McTracerMemsetEvent,
    iter_mctracer_device_events,
    iter_mctracer_trace_events,
)
from dataset_pipeline.conversion.mctracer_to_nsys import stream_kernels
from dataset_pipeline.conversion.mctracer_sync_boundaries import (
    collect_sync_windows,
    write_sync_windows,
)


def _kernel_event():
    return {
        "name": 'kernel {"quoted"}',
        "ph": "X",
        "cat": "0",
        "pid": 2,
        "ts": 101,
        "dur": 17,
        "args": {
            "device_id": 1,
            "grid": {"x": 3, "y": 2, "z": 1},
            "block": {"x": 64, "y": 2, "z": 1},
            "mem": {
                "static_shared": 128,
                "dynamic_shared": 256,
                "registers_per_thread": 37,
                "private_per_thread": 0,
                "private_total": 0,
            },
        },
    }


def _memcpy_event():
    return {
        "name": "an opaque event name",
        "ph": "X",
        "cat": "0",
        "pid": 2,
        "ts": 120,
        "dur": 5,
        "args": {
            "device_id": 1,
            "bytes": 4096,
            "kind": 1,
            "copy_kind": 8,
            "src_kind": 3,
            "dst_kind": 2,
            "chnl_type": 4294967295,
        },
    }


def _memset_event():
    return {
        "name": "another opaque event name",
        "ph": "X",
        "cat": "0",
        "pid": 2,
        "ts": 130,
        "dur": 3,
        "args": {
            "device_id": 1,
            "bytes": 512,
            "kind": 2,
            "mem_kind": 3,
            "value": 0,
            "chnl_type": 4294967295,
        },
    }


def _write_trace(path, events):
    path.write_text(
        json.dumps(
            {
                "displayTimeUnit": "ns",
                "traceEvents": events,
                "metadata": {"nested": [1, {"value": "}"}]},
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _write_raw_event(path, event_text):
    path.write_text(
        '{"displayTimeUnit":"ns","traceEvents":[' + event_text + "]}",
        encoding="utf-8",
    )


def test_streams_typed_device_events_and_ignores_cpu_wrapper(tmp_path):
    cpu_wrapper = {
        "name": "mcLaunchKernel",
        "ph": "X",
        "pid": 1,
        "ts": 99,
        "dur": 1,
        "args": {
            "co_id": 4,
            "link": 0,
            "process_id": 7,
            "ret": 0,
            "thread_id": 8,
        },
    }
    path = tmp_path / "mctracer.json"
    _write_trace(path, [cpu_wrapper, _kernel_event(), _memcpy_event(), _memset_event()])

    events = list(iter_mctracer_device_events(path, chunk_size=7))

    assert events == [
        McTracerKernelEvent(
            event_type="kernel",
            name='kernel {"quoted"}',
            ts_ns=101,
            dur_ns=17,
            device_id=1,
            grid=(3, 2, 1),
            block=(64, 2, 1),
            static_shared_memory_bytes=128,
            dynamic_shared_memory_bytes=256,
            registers_per_thread=37,
        ),
        McTracerMemcpyEvent(
            event_type="memcpy",
            name="an opaque event name",
            ts_ns=120,
            dur_ns=5,
            device_id=1,
            bytes=4096,
            copy_kind=8,
            src_kind=3,
            dst_kind=2,
        ),
        McTracerMemsetEvent(
            event_type="memset",
            name="another opaque event name",
            ts_ns=130,
            dur_ns=3,
            device_id=1,
            bytes=512,
            mem_kind=3,
            value=0,
        ),
    ]


def test_streams_raw_events_with_correlation_metadata(tmp_path):
    cpu_wrapper = {
        "name": "mcGraphLaunch",
        "ph": "X",
        "pid": 1,
        "ts": 99,
        "dur": 1,
        "args": {
            "co_id": 4,
            "link": True,
            "process_id": 7,
            "ret": 0,
            "thread_id": 8,
        },
    }
    kernel = _kernel_event()
    kernel["args"]["co_id"] = 4
    path = tmp_path / "mctracer.json"
    _write_trace(path, [cpu_wrapper, kernel])

    events = list(iter_mctracer_trace_events(path, chunk_size=7))

    assert events == [(0, cpu_wrapper), (1, kernel)]


@pytest.mark.parametrize(
    ("event_factory", "remove_from", "field", "error"),
    [
        (_kernel_event, "event", "dur", KeyError),
        (_kernel_event, "args", "block", KeyError),
        (_kernel_event, "mem", "registers_per_thread", KeyError),
        (_memcpy_event, "args", "copy_kind", KeyError),
        (_memset_event, "args", "mem_kind", KeyError),
    ],
)
def test_typed_device_event_missing_required_field_fails_loud(
    tmp_path, event_factory, remove_from, field, error
):
    event = event_factory()
    if remove_from == "event":
        del event[field]
    elif remove_from == "args":
        del event["args"][field]
    elif remove_from == "mem":
        del event["args"]["mem"][field]
    else:
        raise AssertionError(f"unknown test mutation target: {remove_from}")
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    with pytest.raises(error):
        list(iter_mctracer_device_events(path, chunk_size=5))


def test_unknown_numeric_device_memory_kind_fails_loud(tmp_path):
    event = _memcpy_event()
    event["args"]["kind"] = 99
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    with pytest.raises(ValueError, match="unsupported value 99"):
        list(iter_mctracer_device_events(path, chunk_size=3))

    with pytest.raises(ValueError, match="unsupported value 99"):
        list(stream_kernels(path))


def test_unknown_copy_kind_fails_loud(tmp_path):
    event = _memcpy_event()
    event["args"]["copy_kind"] = 7
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    with pytest.raises(ValueError, match="copy_kind must be one of"):
        list(iter_mctracer_device_events(path))


def test_device_to_device_copy_kind_is_typed(tmp_path):
    event = _memcpy_event()
    event["args"]["copy_kind"] = 3
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [event]}))

    assert list(iter_mctracer_device_events(path))[0].copy_kind == 3


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("src_kind", 1), ("dst_kind", 9)],
)
def test_unknown_copy_endpoint_kind_fails_loud(tmp_path, field, invalid_value):
    event = _memcpy_event()
    event["args"][field] = invalid_value
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    with pytest.raises(ValueError, match=field):
        list(iter_mctracer_device_events(path))


def test_unknown_memset_memory_kind_fails_loud(tmp_path):
    event = _memset_event()
    event["args"]["mem_kind"] = 4
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    with pytest.raises(ValueError, match="mem_kind"):
        list(iter_mctracer_device_events(path))


@pytest.mark.parametrize("field", ["ts", "dur"])
def test_timestamp_and_duration_must_be_integers(tmp_path, field):
    event = _kernel_event()
    event[field] = 1.5
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    with pytest.raises(TypeError, match=field):
        list(iter_mctracer_device_events(path))


@pytest.mark.parametrize(
    "extra",
    [
        {"kind": 1, "bytes": 1},
        {"copy_kind": 1},
        {"mem_kind": 3},
    ],
)
def test_kernel_and_memory_discriminants_are_mutually_exclusive(tmp_path, extra):
    event = _kernel_event()
    event["args"].update(extra)
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    with pytest.raises(ValueError, match="mixes kernel and memory-event"):
        list(iter_mctracer_device_events(path))


@pytest.mark.parametrize(
    "event_factory,extra",
    [(_memcpy_event, {"mem_kind": 3}), (_memset_event, {"copy_kind": 1})],
)
def test_copy_and_memset_discriminants_are_mutually_exclusive(
    tmp_path, event_factory, extra
):
    event = event_factory()
    event["args"].update(extra)
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    with pytest.raises(ValueError, match="mixes memcpy and memset"):
        list(iter_mctracer_device_events(path))


@pytest.mark.parametrize(
    ("original", "duplicate"),
    [
        ('"name":"kernel', '"name":"first","name":"kernel'),
        ('"device_id":1', '"device_id":0,"device_id":1'),
        ('"x":3', '"x":4,"x":3'),
        ('"static_shared":128', '"static_shared":0,"static_shared":128'),
    ],
)
def test_duplicate_keys_at_every_event_object_level_fail_loud(
    tmp_path, original, duplicate
):
    raw_event = json.dumps(_kernel_event(), separators=(",", ":"))
    raw_event = raw_event.replace(original, duplicate, 1)
    path = tmp_path / "mctracer.json"
    _write_raw_event(path, raw_event)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        list(iter_mctracer_device_events(path, chunk_size=3))


def test_duplicate_root_metadata_key_fails_loud(tmp_path):
    path = tmp_path / "mctracer.json"
    path.write_text(
        '{"metadata":{},"metadata":{},"traceEvents":[]}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate JSON object key 'metadata'"):
        list(iter_mctracer_device_events(path, chunk_size=2))


def test_legacy_converter_reuses_typed_parser(tmp_path):
    path = tmp_path / "mctracer.json"
    _write_trace(path, [_memcpy_event(), _kernel_event(), _memset_event()])

    assert list(stream_kernels(path)) == [
        McTracerKernelEvent(
            event_type="kernel",
            name='kernel {"quoted"}',
            ts_ns=101,
            dur_ns=17,
            device_id=1,
            grid=(3, 2, 1),
            block=(64, 2, 1),
            static_shared_memory_bytes=128,
            dynamic_shared_memory_bytes=256,
            registers_per_thread=37,
        )
    ]


def test_collect_sync_windows_prefers_primary_over_stream_events(tmp_path):
    path = tmp_path / "trace.json"
    stream = {
        "name": "mcStreamSynchronize",
        "ph": "X",
        "ts": 10,
        "dur": 2,
        "args": {},
    }
    primary = {
        "name": "mcEventSynchronize",
        "ph": "X",
        "ts": 30,
        "dur": 4,
        "args": {},
    }
    later_primary = {
        "name": "mcDeviceSynchronize",
        "ph": "X",
        "ts": 50,
        "dur": 3,
        "args": {},
    }
    _write_trace(path, [stream, later_primary, primary])

    assert collect_sync_windows(path) == ((30, 34), (50, 53))


def test_write_sync_windows_uses_stream_events_when_primary_is_absent(tmp_path):
    trace_path = tmp_path / "tracer_out-1.json"
    _write_trace(trace_path, [
        {
            "name": "mcStreamSynchronize",
            "ph": "X",
            "ts": 80,
            "dur": 5,
            "args": {},
        },
        {
            "name": "mcStreamSynchronize",
            "ph": "X",
            "ts": 20,
            "dur": 7,
            "args": {},
        },
    ])
    sidecar = tmp_path / ".mctracer_sync_ends_0.json"

    assert write_sync_windows(trace_path, sidecar) == ((20, 27), (80, 85))
    assert json.loads(sidecar.read_text(encoding="utf-8")) == [[20, 27], [80, 85]]


def test_legacy_converter_rebuild_golden_preserves_and_sorts_launches(
    tmp_path, monkeypatch
):
    case_dir = tmp_path / "case"
    trace_dir = case_dir / "mctracer_out"
    trace_dir.mkdir(parents=True)
    first = _kernel_event()
    first["ts"] = 300
    first["dur"] = 19
    second = _kernel_event()
    second["name"] = "earlier_kernel"
    second["ts"] = 200
    second["dur"] = 11
    trace_path = trace_dir / "mctracer-1.json"
    _write_trace(trace_path, [first, _memcpy_event(), second])
    old_payload = {"kernels": [{"name": "old"}]}
    output_path = case_dir / "nsys_kernels_full.json"
    output_path.write_text(json.dumps(old_payload), encoding="utf-8")
    sync_path = case_dir / ".mctracer_sync_ends_1.json"
    sync_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(mctracer_to_nsys.os.path, "getsize", lambda _: 50_000_001)

    assert mctracer_to_nsys.rebuild(case_dir) is True

    assert json.loads(
        (case_dir / "nsys_kernels_full.flat.json.bak").read_text(encoding="utf-8")
    ) == old_payload
    assert not sync_path.exists()
    rebuilt = json.loads(output_path.read_text(encoding="utf-8"))
    assert rebuilt == {
        "kernels": [],
        "per_rank": {
            "1": [
                {
                    "name": "earlier_kernel",
                    "ts_ns": 200,
                    "dur_ns": 11,
                    "grid": [3, 2, 1],
                    "block": [64, 2, 1],
                    "smem": 384,
                    "static_smem": 128,
                    "dynamic_smem": 256,
                    "registers_per_thread": 37,
                },
                {
                    "name": 'kernel {"quoted"}',
                    "ts_ns": 300,
                    "dur_ns": 19,
                    "grid": [3, 2, 1],
                    "block": [64, 2, 1],
                    "smem": 384,
                    "static_smem": 128,
                    "dynamic_smem": 256,
                    "registers_per_thread": 37,
                },
            ]
        },
    }


def test_trace_root_requires_trace_events_array(tmp_path):
    path = tmp_path / "mctracer.json"
    path.write_text('{"traceEvents": {}}', encoding="utf-8")

    with pytest.raises(TypeError, match="traceEvents must be an array"):
        list(iter_mctracer_device_events(path))


def test_truncated_trace_fails_loud(tmp_path):
    raw_event = json.dumps(_kernel_event(), separators=(",", ":"))
    path = tmp_path / "mctracer.json"
    path.write_text('{"traceEvents":[' + raw_event[:-5], encoding="utf-8")

    with pytest.raises(ValueError, match="invalid or truncated"):
        list(iter_mctracer_device_events(path, chunk_size=11))


def test_trailing_trace_data_fails_loud(tmp_path):
    path = tmp_path / "mctracer.json"
    path.write_text('{"traceEvents":[]} trailing', encoding="utf-8")

    with pytest.raises(ValueError, match="trailing data"):
        list(iter_mctracer_device_events(path, chunk_size=4))


def test_single_event_size_is_bounded(tmp_path):
    event = _kernel_event()
    event["name"] = "x" * 256
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    with pytest.raises(ValueError, match="exceeds maximum encoded size"):
        list(
            iter_mctracer_device_events(
                path, chunk_size=64, max_event_bytes=128
            )
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"chunk_size": True}, TypeError, "chunk_size must be an integer"),
        ({"chunk_size": 1.5}, TypeError, "chunk_size must be an integer"),
        (
            {"max_event_bytes": False},
            TypeError,
            "max_value_bytes must be an integer",
        ),
        (
            {"chunk_size": 129, "max_event_bytes": 128},
            ValueError,
            "chunk_size cannot exceed max_value_bytes",
        ),
    ],
)
def test_parser_buffer_limits_are_strict(tmp_path, kwargs, error, message):
    path = tmp_path / "mctracer.json"
    _write_trace(path, [_kernel_event()])

    with pytest.raises(error, match=message):
        list(iter_mctracer_device_events(path, **kwargs))


def test_unicode_event_fields_stream_across_chunks(tmp_path):
    event = _kernel_event()
    event["name"] = "kernel-沐🚀"
    path = tmp_path / "mctracer.json"
    _write_trace(path, [event])

    parsed = list(iter_mctracer_device_events(path, chunk_size=1))

    assert parsed[0].name == "kernel-沐🚀"


def test_bulk_event_stream_has_reasonable_parser_throughput(tmp_path):
    event_count = 20_000
    raw_event = json.dumps(_memcpy_event(), separators=(",", ":"))
    path = tmp_path / "mctracer.json"
    path.write_text(
        '{"traceEvents":[' + ",".join([raw_event] * event_count) + "]}",
        encoding="utf-8",
    )

    started = time.perf_counter()
    parsed_count = sum(1 for _ in iter_mctracer_device_events(path))
    elapsed_seconds = time.perf_counter() - started

    assert parsed_count == event_count
    assert elapsed_seconds < 5.0
