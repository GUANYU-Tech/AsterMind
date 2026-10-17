import math

import pytest

from evaluation.kernel.device_event import (
    measure_current_stream_events,
    measure_differential_rotating_graph_replay_events,
    measure_differential_rotating_npu_graph_replay_events,
    measure_graph_replay_events,
    measure_rotating_graph_replay_events,
    prepare_bf16_cache_eviction_sweep,
    summarize_event_samples,
    validate_cache_eviction_mib,
)


class _FakeEvent:
    _next_timestamp_ms = 0.0

    def __init__(self, *, enable_timing):
        assert enable_timing is True
        self.timestamp_ms = None

    def record(self, stream):
        assert stream is _FakeDevice.stream
        self.timestamp_ms = self.__class__._next_timestamp_ms
        self.__class__._next_timestamp_ms += 0.000001

    def synchronize(self):
        if self.timestamp_ms is None:
            raise AssertionError("cannot synchronize an unrecorded Event")

    def elapsed_time(self, end):
        return end.timestamp_ms - self.timestamp_ms


class _FakeDevice:
    Event = _FakeEvent
    stream = object()

    @staticmethod
    def current_stream():
        return _FakeDevice.stream

    @staticmethod
    def synchronize():
        return None


class _OrderingEvent:
    trace = []
    created_count = 0
    next_timestamp_ms = 0.0

    def __init__(self, *, enable_timing):
        assert enable_timing is True
        self.label = (
            "start" if self.__class__.created_count % 2 == 0 else "end")
        self.__class__.created_count += 1
        self.timestamp_ms = None
        self.__class__.trace.append(f"{self.label}.create")

    def record(self, stream):
        assert stream is _OrderingDevice.stream
        self.__class__.trace.append(f"{self.label}.record")
        self.timestamp_ms = self.__class__.next_timestamp_ms
        self.__class__.next_timestamp_ms += 0.000001

    def synchronize(self):
        self.__class__.trace.append(f"{self.label}.synchronize")

    def elapsed_time(self, end):
        self.__class__.trace.append(f"{self.label}.elapsed_time")
        return end.timestamp_ms - self.timestamp_ms


class _OrderingDevice:
    Event = _OrderingEvent
    stream = object()

    @staticmethod
    def current_stream():
        _OrderingEvent.trace.append("current_stream")
        return _OrderingDevice.stream

    @staticmethod
    def synchronize():
        _OrderingEvent.trace.append("device.synchronize")


class _FakeGraph:
    def __init__(self):
        self.operation_count = 0

    def replay(self):
        _FakeEvent._next_timestamp_ms += self.operation_count * 0.000001


class _FakeGraphContext:
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        _FakeGraphDevice.capturing = self.graph

    def __exit__(self, exc_type, exc_value, traceback):
        _FakeGraphDevice.capturing = None


class _FakeGraphDevice(_FakeDevice):
    CUDAGraph = _FakeGraph
    NPUGraph = _FakeGraph
    capturing = None

    @staticmethod
    def graph(graph):
        return _FakeGraphContext(graph)


class _FakeSweepTensor:
    def __init__(self, element_count):
        self.element_count = element_count
        self.add_values = []
        self.fill_values = []

    def numel(self):
        return self.element_count

    def element_size(self):
        return 2

    def add_(self, value):
        self.add_values.append(value)
        return self

    def fill_(self, value):
        self.fill_values.append(value)
        return self


class _FakeTorch:
    bfloat16 = object()
    allocations = []

    @classmethod
    def zeros(cls, element_count, *, dtype, device):
        assert dtype is cls.bfloat16
        tensor = _FakeSweepTensor(element_count)
        cls.allocations.append((tensor, device))
        return tensor


class _OrderingGraph:
    def __init__(self):
        self.operation_count = 0
        _OrderingEvent.trace.append("graph.create")

    def replay(self):
        _OrderingEvent.trace.append("graph.replay")
        _OrderingEvent.next_timestamp_ms += (
            0.000010 + self.operation_count * 0.000001)


class _OrderingGraphContext:
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        _OrderingEvent.trace.append("graph.capture.enter")
        _OrderingGraphDevice.capturing = self.graph

    def __exit__(self, exc_type, exc_value, traceback):
        _OrderingGraphDevice.capturing = None
        _OrderingEvent.trace.append("graph.capture.exit")


class _OrderingGraphDevice(_OrderingDevice):
    CUDAGraph = _OrderingGraph
    NPUGraph = _OrderingGraph
    capturing = None

    @staticmethod
    def graph(graph):
        return _OrderingGraphContext(graph)


def test_event_sample_statistics_report_required_distribution():
    summary = summarize_event_samples([1.0, 2.0, 3.0, 4.0, 10.0])

    assert summary["median_ns"] == 3.0
    assert summary["p10_ns"] == pytest.approx(1.4)
    assert summary["p90_ns"] == pytest.approx(7.6)
    assert summary["standard_deviation_ns"] == pytest.approx(
        math.sqrt(10.0))
    assert summary["coefficient_of_variation"] == pytest.approx(
        math.sqrt(10.0) / 3.0)


def test_event_measurement_keeps_warmup_outside_independent_samples():
    _FakeEvent._next_timestamp_ms = 0.0
    calls = []

    result = measure_current_stream_events(
        lambda: calls.append(len(calls)),
        device_module=_FakeDevice,
        warmup_iterations=3,
        sample_count=4,
    )

    assert len(calls) == 8
    assert result["schema_version"] == 3
    assert result["warmup_iterations"] == 3
    assert result["event_prelude_iterations"] == 1
    assert result["sample_count"] == 4
    assert [row["sample_index"] for row in result["samples"]] == [0, 1, 2, 3]
    assert [row["event_duration_ns"] for row in result["samples"]] == (
        pytest.approx([1.0, 1.0, 1.0, 1.0]))


def test_event_measurement_accepts_and_releases_exact_op_return_value():
    result = measure_current_stream_events(
        lambda: object(),
        device_module=_FakeDevice,
        warmup_iterations=1,
        sample_count=1,
    )

    assert result["sample_count"] == 1


def test_event_measurement_batches_distinct_public_calls_per_sample():
    _FakeEvent._next_timestamp_ms = 0.0
    calls = []

    result = measure_current_stream_events(
        lambda: calls.append(len(calls)),
        device_module=_FakeDevice,
        warmup_iterations=2,
        sample_count=3,
        operation_repetitions_per_sample=512,
    )

    assert len(calls) == (2 + 1 + 3) * 512
    assert result["schema_version"] == 4
    assert result["operation_repetitions_per_sample"] == 512
    assert [sample["operation_repetition_count"] for sample in result[
        "samples"]] == [512, 512, 512]
    assert [sample["event_duration_ns"] for sample in result["samples"]] == (
        pytest.approx([1.0 / 512.0] * 3))
    assert all(
        sample["aggregate_event_duration_ns"] == pytest.approx(1.0)
        for sample in result["samples"])


def test_event_measurement_runs_pre_event_operation_outside_every_interval():
    _FakeEvent._next_timestamp_ms = 0.0
    calls = []

    result = measure_current_stream_events(
        lambda: calls.append("target"),
        device_module=_FakeDevice,
        warmup_iterations=2,
        sample_count=3,
        pre_event_operation=lambda: calls.append("power"),
    )

    assert calls == [
        "power", "target", "power", "target", "power", "target",
        "power", "target", "power", "target", "power", "target",
    ]
    assert [sample["event_duration_ns"] for sample in result["samples"]] == (
        pytest.approx([1.0, 1.0, 1.0]))


def test_bf16_cache_eviction_sweep_allocates_exact_bytes_and_retains_buffer():
    _FakeTorch.allocations = []

    sweep, actual_bytes = prepare_bf16_cache_eviction_sweep(
        torch_module=_FakeTorch, device="cuda", cache_eviction_mib=2048)

    assert actual_bytes == 2048 * 1024 * 1024
    assert len(_FakeTorch.allocations) == 1
    tensor, device = _FakeTorch.allocations[0]
    assert device == "cuda"
    assert sweep() is tensor
    assert tensor.add_values == [1]


def test_bf16_cache_eviction_sweep_can_use_non_add_fill_symbol():
    _FakeTorch.allocations = []

    sweep, actual_bytes = prepare_bf16_cache_eviction_sweep(
        torch_module=_FakeTorch, device="npu", cache_eviction_mib=2048,
        sweep_operation="fill")

    assert actual_bytes == 2048 * 1024 * 1024
    tensor, device = _FakeTorch.allocations[0]
    assert device == "npu"
    assert sweep() is tensor
    assert tensor.fill_values == [1]
    assert tensor.add_values == []


def test_bf16_cache_eviction_sweep_zero_disables_allocation():
    _FakeTorch.allocations = []

    assert prepare_bf16_cache_eviction_sweep(
        torch_module=_FakeTorch, device="npu", cache_eviction_mib=0,
    ) == (None, 0)
    assert _FakeTorch.allocations == []


@pytest.mark.parametrize("value", (-1, True, 1.5, "1"))
def test_bf16_cache_eviction_sweep_rejects_invalid_mib(value):
    with pytest.raises((TypeError, ValueError)):
        prepare_bf16_cache_eviction_sweep(
            torch_module=_FakeTorch, device="cuda", cache_eviction_mib=value)


def test_bf16_cache_eviction_sweep_rejects_unknown_operation():
    with pytest.raises(ValueError, match="sweep_operation"):
        prepare_bf16_cache_eviction_sweep(
            torch_module=_FakeTorch, device="cuda", cache_eviction_mib=2048,
            sweep_operation="mul")


@pytest.mark.parametrize(
    "value,minimum", ((0, 320), (320, 320), (2048, 1536)))
def test_cache_eviction_threshold_accepts_warm_or_at_least_eight_l2(
        value, minimum):
    validate_cache_eviction_mib(
        value, minimum_nonzero_mib=minimum, hardware_id="test_device")


@pytest.mark.parametrize("value,minimum", ((319, 320), (1535, 1536), (63, 64)))
def test_cache_eviction_threshold_rejects_small_nonzero_sweep(
        value, minimum):
    with pytest.raises(ValueError, match="8x L2"):
        validate_cache_eviction_mib(
            value, minimum_nonzero_mib=minimum, hardware_id="test_device")


def test_event_measurement_joins_internal_streams_before_end_event():
    _FakeEvent._next_timestamp_ms = 0.0
    order = []

    measure_current_stream_events(
        lambda: order.append("operation"),
        device_module=_FakeDevice,
        warmup_iterations=1,
        sample_count=2,
        join_operation_streams=lambda: order.append("join"),
    )

    assert order == [
        "operation", "operation", "join", "operation", "join",
        "operation", "join",
    ]


def test_event_measurement_evicts_cache_outside_each_timed_sample():
    _FakeEvent._next_timestamp_ms = 0.0
    order = []

    result = measure_current_stream_events(
        lambda: order.append("operation"),
        device_module=_FakeDevice,
        warmup_iterations=2,
        sample_count=3,
        cache_eviction_operation=lambda: order.append("evict"),
        cache_eviction_bytes=256 * 1024 * 1024,
    )

    assert order == [
        "evict", "operation", "evict", "operation", "evict", "operation",
        "evict", "operation", "evict", "operation", "evict", "operation",
    ]
    assert result["event_prelude_iterations"] == 1
    assert result["cache_eviction_requested_bytes"] == 256 * 1024 * 1024
    assert result["cache_eviction_bytes_before_sample"] == 256 * 1024 * 1024
    assert result["statistics"]["median_ns"] == pytest.approx(1.0)


def test_event_measurement_runs_full_event_prelude_before_samples():
    _OrderingEvent.trace = []
    _OrderingEvent.created_count = 0
    _OrderingEvent.next_timestamp_ms = 0.0

    result = measure_current_stream_events(
        lambda: _OrderingEvent.trace.append("operation"),
        device_module=_OrderingDevice,
        warmup_iterations=1,
        sample_count=1,
        join_operation_streams=lambda: _OrderingEvent.trace.append("join"),
        cache_eviction_operation=(
            lambda: _OrderingEvent.trace.append("evict")),
        cache_eviction_bytes=256 * 1024 * 1024,
    )

    assert _OrderingEvent.trace == [
        "current_stream",
        "device.synchronize",
        "evict",
        "operation",
        "device.synchronize",
        "start.create",
        "end.create",
        "evict",
        "start.record",
        "operation",
        "join",
        "end.record",
        "end.synchronize",
        "device.synchronize",
        "start.elapsed_time",
        "start.create",
        "end.create",
        "evict",
        "start.record",
        "operation",
        "join",
        "end.record",
        "end.synchronize",
        "device.synchronize",
        "start.elapsed_time",
    ]
    assert result["event_prelude_iterations"] == 1
    assert result["sample_count"] == 1
    assert len(result["samples"]) == 1
    assert result["samples"][0]["sample_index"] == 0


def test_graph_replay_measurement_reports_per_call_time_and_convergence():
    _FakeEvent._next_timestamp_ms = 0.0
    calls = []

    def operation():
        calls.append("operation")
        if _FakeGraphDevice.capturing is not None:
            _FakeGraphDevice.capturing.operation_count += 1

    result = measure_graph_replay_events(
        operation,
        device_module=_FakeGraphDevice,
        warmup_iterations=2,
        sample_count=3,
        repetition_counts=(10, 20),
    )

    assert len(calls) == 32
    assert result["schema_version"] == 3
    assert result["mode"] == "external_events_around_cuda_graph_replay"
    assert result["event_prelude_iterations_per_repetition"] == 1
    assert result["cache_eviction_requested_bytes"] == 0
    assert result["cache_eviction_bytes_before_replay"] == 0
    assert result["repetition_counts"] == [10, 20]
    assert result["statistics"]["median_ns"] == pytest.approx(1.05)


def test_graph_replay_cache_sweep_is_outside_capture_and_event_interval():
    _OrderingEvent.trace = []
    _OrderingEvent.created_count = 0
    _OrderingEvent.next_timestamp_ms = 0.0

    result = measure_graph_replay_events(
        lambda: _OrderingEvent.trace.append("operation"),
        device_module=_OrderingGraphDevice,
        warmup_iterations=1,
        sample_count=1,
        repetition_counts=(2, 3),
        cache_eviction_operation=lambda: _OrderingEvent.trace.append("evict"),
        cache_eviction_bytes=256 * 1024 * 1024,
    )

    trace = _OrderingEvent.trace
    timed_eviction_indexes = [
        index for index, value in enumerate(trace[:-1])
        if value == "evict" and trace[index + 1] == "start.record"]
    assert len(timed_eviction_indexes) == 4
    for eviction_index in timed_eviction_indexes:
        preceding_capture_enter = max(
            index for index, value in enumerate(trace[:eviction_index])
            if value == "graph.capture.enter")
        preceding_capture_exit = max(
            index for index, value in enumerate(trace[:eviction_index])
            if value == "graph.capture.exit")
        assert preceding_capture_exit > preceding_capture_enter
    assert result["schema_version"] == 3
    assert result["event_prelude_iterations_per_repetition"] == 1
    assert result["cache_eviction_requested_bytes"] == 256 * 1024 * 1024
    assert result["cache_eviction_bytes_before_replay"] == 256 * 1024 * 1024
    assert result["convergence"]["status"] == "converged"


def test_rotating_graph_replay_cycles_distinct_operations():
    _FakeEvent._next_timestamp_ms = 0.0
    calls = []

    def operation(label):
        def invoke():
            calls.append(label)
            if _FakeGraphDevice.capturing is not None:
                _FakeGraphDevice.capturing.operation_count += 1
        return invoke

    result = measure_rotating_graph_replay_events(
        (operation("a"), operation("b"), operation("c")),
        device_module=_FakeGraphDevice,
        warmup_iterations=2,
        sample_count=3,
        repetition_counts=(10, 20),
    )

    assert calls[:12] == ["a", "b", "a", "b", "c", "a", "b", "c",
                          "a", "b", "c", "a"]
    assert result["operation_cycle_length"] == 3
    assert result["statistics"]["median_ns"] == pytest.approx(1.05)


def test_differential_graph_replay_cancels_fixed_event_cost_and_alternates():
    _FakeEvent._next_timestamp_ms = 0.0
    calls = []

    def operation(index):
        def invoke():
            calls.append(index)
            if _FakeGraphDevice.capturing is not None:
                _FakeGraphDevice.capturing.operation_count += 1
        return invoke

    result = measure_differential_rotating_graph_replay_events(
        tuple(operation(index) for index in range(8)),
        device_module=_FakeGraphDevice,
        warmup_iterations=2,
        sample_count=4,
        repetition_counts=(4, 8),
        per_call_distinct_input_bytes=16,
        minimum_distinct_input_bytes=64,
        hardware_l2_bytes=8,
    )

    assert result["mode"] == (
        "paired_differential_events_around_cuda_graph_replay")
    assert result["repetition_counts"] == [4, 8]
    assert result["operation_cycle_length"] == 8
    assert result["distinct_input_contract"]["lower_replay_bytes"] == 64
    assert result["distinct_input_contract"]["higher_replay_bytes"] == 128
    assert result["distinct_input_contract"]["delta_replay_bytes"] == 64
    assert result["distinct_input_contract"][
        "configured_minimum_l2_multiple"] == 8
    assert [sample["replay_order"] for sample in result["samples"]] == [
        "lower_then_higher", "higher_then_lower",
        "lower_then_higher", "higher_then_lower",
    ]
    assert [sample["event_duration_ns"] for sample in result["samples"]] == (
        pytest.approx([1.0, 1.0, 1.0, 1.0]))
    assert result["statistics"]["median_ns"] == pytest.approx(1.0)
    assert result["repetition_runs"][0]["samples"][0][
        "replay_duration_ns"] == pytest.approx(5.0)
    assert result["repetition_runs"][1]["samples"][0][
        "replay_duration_ns"] == pytest.approx(9.0)


def test_differential_graph_replay_conditions_outside_every_replay():
    _OrderingEvent.trace = []
    _OrderingEvent.created_count = 0
    _OrderingEvent.next_timestamp_ms = 0.0

    def operation():
        _OrderingEvent.trace.append("operation")
        if _OrderingGraphDevice.capturing is not None:
            _OrderingGraphDevice.capturing.operation_count += 1

    result = measure_differential_rotating_graph_replay_events(
        tuple(operation for _ in range(4)),
        device_module=_OrderingGraphDevice,
        warmup_iterations=1,
        sample_count=2,
        repetition_counts=(2, 4),
        per_call_distinct_input_bytes=32,
        minimum_distinct_input_bytes=64,
        hardware_l2_bytes=8,
        cache_eviction_operation=lambda: _OrderingEvent.trace.append("evict"),
        cache_eviction_bytes=256,
        pre_event_operation=lambda: _OrderingEvent.trace.append("power"),
    )

    trace = _OrderingEvent.trace
    for index, value in enumerate(trace):
        if value == "graph.replay":
            assert "evict" in trace[:index]
    timed_replays = [
        index for index, value in enumerate(trace)
        if value == "graph.replay" and index > 0
        and trace[index - 1] == "start.record"]
    assert len(timed_replays) == 6
    for replay_index in timed_replays:
        assert trace[replay_index - 2] == "power"
        assert trace[replay_index - 3] == "evict"
    assert result["pre_event_operation_executions_per_replay"] == 1


class _FakeStorage:
    def __init__(self, address, size):
        self.address = address
        self.size = size

    def data_ptr(self):
        return self.address

    def nbytes(self):
        return self.size


class _FakeTensor:
    device = "npu:0"

    def __init__(self, address, size=32):
        self.storage = _FakeStorage(address, size)

    def untyped_storage(self):
        return self.storage


def test_npu_differential_graph_uses_fixed_counts_and_large_sweep():
    _FakeEvent._next_timestamp_ms = 0.0
    tensors = tuple(_FakeTensor(index + 1) for index in range(256))
    outputs = tuple(_FakeTensor(index + 257) for index in range(256))

    def operation(index):
        def invoke():
            if _FakeGraphDevice.capturing is not None:
                _FakeGraphDevice.capturing.operation_count += 1
            return outputs[index]
        return invoke

    result = measure_differential_rotating_npu_graph_replay_events(
        tuple(operation(index) for index in range(256)),
        operation_storage_groups=tuple((tensor,) for tensor in tensors),
        device_module=_FakeGraphDevice,
        warmup_iterations=1,
        sample_count=2,
        hardware_l2_bytes=192 * 1024 * 1024,
        cache_eviction_operation=lambda: None,
        cache_eviction_bytes=2 * 1024 * 1024 * 1024,
    )

    assert result["repetition_counts"] == [16, 256]
    assert [sample["replay_order"] for sample in result["samples"]] == [
        "lower_then_higher", "higher_then_lower"]
    contract = result["rotating_buffer_contract"]
    assert contract["cache_eviction_l2_multiple"] == pytest.approx(32 / 3)
    assert contract["cache_eviction_l2_multiple_rounded_2dp"] == 10.67
    assert contract["preallocated_distinct_bytes"] == 256 * 32
    assert contract["graph_output_distinct_bytes"] == 256 * 32
    assert contract["rotation_total_distinct_bytes"] == 512 * 32
    assert contract["addresses_unique_within_each_graph"] is True


def test_npu_differential_graph_accepts_memory_bounded_dynamic_counts():
    _FakeEvent._next_timestamp_ms = 0.0
    tensors = tuple(_FakeTensor(index + 1) for index in range(32))
    outputs = tuple(_FakeTensor(index + 33) for index in range(32))

    def operation(index):
        def invoke():
            if _FakeGraphDevice.capturing is not None:
                _FakeGraphDevice.capturing.operation_count += 1
            return outputs[index]
        return invoke

    result = measure_differential_rotating_npu_graph_replay_events(
        tuple(operation(index) for index in range(32)),
        operation_storage_groups=tuple((tensor,) for tensor in tensors),
        device_module=_FakeGraphDevice,
        warmup_iterations=1,
        sample_count=2,
        hardware_l2_bytes=192 * 1024 * 1024,
        cache_eviction_operation=lambda: None,
        cache_eviction_bytes=2 * 1024 * 1024 * 1024,
        repetition_counts=(2, 32),
        repetition_selection_rule=(
            "memory_bounded_unique_preallocated_calls_v1"),
    )

    assert result["repetition_counts"] == [2, 32]
    assert result["operation_cycle_length"] == 32
    assert all(sample["repetition_delta"] == 30
               for sample in result["samples"])
    contract = result["rotating_buffer_contract"]
    assert contract["lower_operation_count"] == 2
    assert contract["higher_operation_count"] == 32
    assert contract["selection_rule"] == (
        "memory_bounded_unique_preallocated_calls_v1")


@pytest.mark.parametrize(
    "repetitions,operation_count",
    (((3, 8), 8), ((4, 7), 8), ((4, 8), 7)),
)
def test_differential_graph_replay_rejects_noncanonical_input_coverage(
        repetitions, operation_count):
    with pytest.raises(ValueError):
        measure_differential_rotating_graph_replay_events(
            tuple(lambda: None for _ in range(operation_count)),
            device_module=_FakeGraphDevice,
            warmup_iterations=1,
            sample_count=1,
            repetition_counts=repetitions,
            per_call_distinct_input_bytes=16,
            minimum_distinct_input_bytes=64,
            hardware_l2_bytes=8,
        )


@pytest.mark.parametrize("samples", ([], [0.0], [math.inf], [True]))
def test_event_sample_statistics_reject_invalid_samples(samples):
    with pytest.raises((TypeError, ValueError)):
        summarize_event_samples(samples)
