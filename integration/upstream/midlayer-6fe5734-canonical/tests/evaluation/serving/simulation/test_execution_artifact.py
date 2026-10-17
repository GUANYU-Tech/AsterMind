from dataclasses import replace
import hashlib
import json

import pytest

from evaluation.serving.simulation.execution_artifact import (
    execution_artifact_manifest,
    performance_timing_status,
    _validate_global_reserved_resource_intervals,
)
from src.kernel.kernel_spec import (
    KernelKind,
    KernelPrediction,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.kernel.launch_config import LaunchConfig
from src.serving.kernel_lowering.execution_timeline import (
    ExecutionInterval,
    ExecutionTimeline,
    HostExecutionInterval,
)
from src.serving.kernel_lowering.kernel_plan import KernelCall, KernelPlan
from src.serving.metrics.serving_metrics import ServingMetrics
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduleStep,
    ScheduledRequest,
)
from src.serving.schedule.schedule_trace import (
    RouteDecision,
    ScheduleTrace,
    ScheduleTraceEntry,
    TransferTraceEntry,
)
from src.serving.simulation import (
    SimulationResult,
    SimulationStep,
    SimulationTransfer,
)
from src.serving.workload.workload_graph import (
    WorkloadEdge,
    WorkloadGraph,
    WorkloadKind,
    WorkloadNode,
)


_TIMING_STATUS = "unsupported_a800_analytical_launch"


def _artifact(result: SimulationResult):
    return execution_artifact_manifest(
        result,
        performance_timing_status=_TIMING_STATUS,
    )


def _result() -> SimulationResult:
    scheduled = ScheduledRequest(
        "request-0",
        "decode",
        0,
        1,
        4,
        raw_model_output_tokens=1,
    )
    backend_step = BackendStep(
        ScheduleStep(
            "main:vllm:0",
            "main",
            (scheduled,),
            finished_request_ids=("request-0",),
            attention_kv_layout="paged_varlen",
        ),
        generated_tokens={"request-0": 1},
    )
    workload = WorkloadGraph(
        (
            WorkloadNode(
                "vector",
                WorkloadKind.ELEMENTWISE,
                "main",
                {"element_count": 32},
                dtype="bf16",
                variant="elementwise",
                rank=0,
                attributes={
                    "reads": 1,
                    "writes": 1,
                    "operation": "copy",
                },
            ),
            WorkloadNode(
                "communication",
                WorkloadKind.P2P,
                "main",
                {},
                rank=0,
                attributes={
                    "algorithm": "send_recv",
                    "bytes": 1024,
                    "bytes_semantics": "payload",
                    "backend": "analytical_roofline_v1",
                    "backend_config_sha256": None,
                    "direction": "point_to_point",
                    "link": "nvlink",
                    "topology": "point_to_point",
                    "participants": (0, 1),
                    "operation": "send_recv",
                    "source_rank": 0,
                    "destination_rank": 1,
                    "ring_order": (),
                    "reduction": None,
                    "resource_id": "nvlink",
                },
            ),
        ),
        (WorkloadEdge("vector", "communication"),),
    )
    vector_spec = KernelSpec(
        KernelKind.VECTOR,
        "nvidia_a800",
        shape={"element_count": 32},
        dtype="bf16",
        variant="elementwise",
        launch=LaunchConfig(
            "manual_vector",
            (1, 1, 1),
            (32, 1, 1),
            execution_variant="elementwise",
        ),
        attributes={"reads": 1, "writes": 1, "operation": "copy"},
    )
    communication_spec = KernelSpec(
        KernelKind.COMMUNICATION,
        "nvidia_a800",
        transfer=PhysicalTransferConfig(
            "send_recv",
            1024,
            "point_to_point",
            "nvlink",
            topology="point_to_point",
            participants=(0, 1),
            operation="send_recv",
            bytes_semantics="payload",
            source_rank=0,
            destination_rank=1,
            backend="analytical_roofline_v1",
            resource_id="nvlink",
        ),
    )
    plan = KernelPlan((
        KernelCall(
            "vector", vector_spec, "main", 0, "compute", ("vector",)),
        KernelCall(
            "communication", communication_spec, "main", 0, "nvlink",
            ("communication",),
            dependencies=("vector",)),
    ))
    predictions = {
        "vector": KernelPrediction(
            0.01,
            {"compute": 0.009},
            0.001,
            "compute",
            {"vector": 0.009},
        ),
        "communication": KernelPrediction(
            0.02,
            {"nvlink": 0.019},
            0.001,
            "nvlink",
            {"nvlink": 0.019},
        ),
    }
    timeline = ExecutionTimeline(
        (
            ExecutionInterval(
                "vector", "main", 0, "compute", 0.0, 0.01,
                ("pool:main:rank:0:compute",)),
            ExecutionInterval(
                "communication", "main", 0, "nvlink", 0.01, 0.03,
                (
                    "communication:endpoint:nvidia_a800:0:nvlink",
                    "communication:endpoint:nvidia_a800:1:nvlink",
                    "communication:fabric:nvlink",
                )),
        ),
        0.03,
    )
    step = SimulationStep(
        backend_step,
        workload,
        plan,
        predictions,
        timeline,
    )
    trace = ScheduleTrace((
        ScheduleTraceEntry(backend_step, 1.0, 1.03),
    ))
    return SimulationResult(
        trace,
        ServingMetrics({}, 0.0, 0.0, 0.0),
        steps=(step,),
    )


def _result_with_pd_transfer() -> SimulationResult:
    result = _result()
    base_step = result.steps[0]

    def timeline_for_pool(pool_name):
        intervals = []
        for interval in base_step.execution_timeline.intervals:
            if interval.resource == "compute":
                reserved_resource_ids = (
                    f"pool:{pool_name}:rank:{interval.rank}:compute",)
            elif interval.resource == "nvlink":
                reserved_resource_ids = interval.reserved_resource_ids
            else:
                raise AssertionError(
                    f"unexpected fixture resource: {interval.resource!r}")
            intervals.append(replace(
                interval,
                pool_name=pool_name,
                reserved_resource_ids=reserved_resource_ids,
            ))
        return ExecutionTimeline(
            tuple(intervals),
            base_step.execution_timeline.duration_s,
        )

    source_request = ScheduledRequest(
        "request-0",
        "prefill",
        1,
        0,
        raw_model_output_tokens=1,
    )
    source_backend_step = BackendStep(
        ScheduleStep(
            "prefill:vllm:0",
            "prefill",
            (source_request,),
            finished_request_ids=("request-0",),
            attention_kv_layout="paged_varlen",
        ),
        generated_tokens={"request-0": 1},
    )
    source_workload = WorkloadGraph(
        tuple(replace(node, pool_name="prefill")
              for node in base_step.workload.nodes),
        base_step.workload.edges,
    )
    source_plan = KernelPlan(tuple(
        replace(call, pool_name="prefill")
        for call in base_step.kernel_plan.calls
    ))
    source_timeline = timeline_for_pool("prefill")
    source_step = replace(
        base_step,
        backend_step=source_backend_step,
        workload=source_workload,
        kernel_plan=source_plan,
        execution_timeline=source_timeline,
    )
    destination_request = ScheduledRequest(
        "request-0",
        "decode",
        0,
        1,
        computed_tokens=1,
        raw_model_output_tokens=1,
    )
    destination_backend_step = BackendStep(
        ScheduleStep(
            "decode:vllm:0",
            "decode",
            (destination_request,),
            finished_request_ids=("request-0",),
            attention_kv_layout="paged_varlen",
        ),
        generated_tokens={"request-0": 1},
    )
    destination_workload = WorkloadGraph(
        tuple(replace(node, pool_name="decode")
              for node in base_step.workload.nodes),
        base_step.workload.edges,
    )
    destination_plan = KernelPlan(tuple(
        replace(call, pool_name="decode")
        for call in base_step.kernel_plan.calls
    ))
    destination_timeline = timeline_for_pool("decode")
    destination_step = replace(
        base_step,
        backend_step=destination_backend_step,
        workload=destination_workload,
        kernel_plan=destination_plan,
        execution_timeline=destination_timeline,
    )
    node = WorkloadNode(
        "request-0:pd-kv-transfer:prefill-to-decode",
        WorkloadKind.PD_KV_TRANSFER,
        "prefill",
        {},
        rank=0,
        attributes={
            "algorithm": "send_recv",
            "bytes": 2048,
            "bytes_semantics": "payload",
            "backend": "analytical_roofline_v1",
            "backend_config_sha256": None,
            "direction": "point_to_point",
            "link": "nvlink",
            "topology": "point_to_point",
            "participants": (0, 1),
            "operation": "send_recv",
            "source_rank": 0,
            "destination_rank": 1,
            "ring_order": (),
            "reduction": None,
            "resource_id": "nvlink",
            "destination_pool": "decode",
        },
    )
    workload = WorkloadGraph((node,))
    spec = KernelSpec(
        KernelKind.COMMUNICATION,
        "nvidia_a800",
        transfer=PhysicalTransferConfig(
            "send_recv",
            2048,
            "point_to_point",
            "nvlink",
            topology="point_to_point",
            participants=(0, 1),
            operation="send_recv",
            bytes_semantics="payload",
            source_rank=0,
            destination_rank=1,
            backend="analytical_roofline_v1",
            resource_id="nvlink",
        ),
    )
    call = KernelCall(
        node.node_id,
        spec,
        "prefill",
        0,
        "nvlink",
        (node.node_id,),
    )
    plan = KernelPlan((call,))
    prediction = KernelPrediction(
        0.02,
        {"nvlink": 0.019},
        0.001,
        "nvlink",
        {"nvlink": 0.019},
    )
    timeline = ExecutionTimeline((
        ExecutionInterval(
            call.call_id, "prefill", 0, "nvlink", 0.0, 0.02,
            (
                "communication:endpoint:nvidia_a800:0:nvlink",
                "communication:endpoint:nvidia_a800:1:nvlink",
                "communication:fabric:nvlink",
            )),
    ), 0.02)
    simulation_transfer = SimulationTransfer(
        "request-0",
        "prefill",
        "decode",
        workload,
        plan,
        {call.call_id: prediction},
        timeline,
    )
    trace_transfer = TransferTraceEntry(
        transfer_id=node.node_id,
        request_ids=("request-0",),
        payload="kv",
        source_pool_id="prefill",
        destination_pool_id="decode",
        resource_id="nvlink",
        link="nvlink",
        reserved_resource_ids=(
            "communication:endpoint:nvidia_a800:0:nvlink",
            "communication:endpoint:nvidia_a800:1:nvlink",
            "communication:fabric:nvlink",
        ),
        byte_count=2048,
        start_time_s=1.03,
        end_time_s=1.05,
    )
    route_decision = RouteDecision(
        "deployment",
        "request-0",
        "prefill_to_decode",
        "least_queued_tokens",
        "prefill",
        (("decode", 0),),
        "decode",
        "minimum_visible_queued_tokens_then_engine_id",
        1.03,
        1.03,
        1.03,
        1.03,
    )
    trace = ScheduleTrace(
        (
            ScheduleTraceEntry(source_backend_step, 1.0, 1.03),
            ScheduleTraceEntry(destination_backend_step, 1.05, 1.08),
        ),
        route_decisions=(route_decision,),
        transfers=(trace_transfer,),
    )
    return replace(
        result,
        schedule_trace=trace,
        steps=(source_step, destination_step),
        transfers=(simulation_transfer,),
    )


def test_execution_artifact_serializes_every_layer_and_hashes_canonically():
    manifest = _artifact(_result())

    assert manifest["schema_version"] == 5
    assert manifest["performance_timing_status"] == _TIMING_STATUS
    assert manifest["schedule_trace"]["route_decision_version"] == 2
    assert manifest["schedule_trace"]["transfer_trace_version"] == 2
    assert manifest["counts"] == {
        "schedule_steps": 1,
        "simulation_steps": 1,
        "simulation_transfers": 0,
        "trace_transfers": 0,
        "modeled_kv_offload_transfers": 0,
        "kernel_calls": 2,
    }
    assert manifest["modeled_kv_offload"] == {
        "schema_version": 2,
        "availability": "not_enabled",
        "intent_source": None,
        "transfers": (),
        "final_tier_state": None,
    }
    step = manifest["steps"][0]
    assert step["step_id"] == "main:vllm:0"
    assert step["execution_timeline"]["schema_version"] == 2

    assert step["schedule_interval"] == {
        "start_time_s": 1.0,
        "end_time_s": 1.03,
    }
    assert step["backend_step"]["schedule"]["requests"][0][
        "request_id"] == "request-0"
    assert step["workload_graph"]["topological_order"] == (
        "vector", "communication")
    calls = step["kernel_plan"]["calls"]
    assert tuple(call["call_id"] for call in calls) == (
        "vector", "communication")
    assert calls[0]["kernel_spec"]["launch"] == {
        "algorithm": "manual_vector",
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "tiling": None,
        "cluster": None,
        "parameters": {},
        "execution_variant": "elementwise",
    }
    assert calls[1]["kernel_spec"]["transfer"] == {
        "algorithm": "send_recv",
        "bytes": 1024,
        "direction": "point_to_point",
        "link": "nvlink",
        "topology": "point_to_point",
        "tier": None,
        "participants": (0, 1),
        "operation": "send_recv",
        "bytes_semantics": "payload",
        "source_rank": 0,
        "destination_rank": 1,
        "ring_order": (),
        "reduction": None,
        "backend": "analytical_roofline_v1",
        "resource_id": "nvlink",
        "backend_config_sha256": None,
    }
    assert tuple(
        prediction["call_id"] for prediction in step["predictions"]
    ) == ("vector", "communication")
    assert tuple(
        interval["call_id"]
        for interval in step["execution_timeline"]["intervals"]
    ) == ("vector", "communication")
    assert step["execution_timeline"]["origin_time_s"] == 1.0
    assert step["execution_timeline"]["intervals"][0][
        "absolute_start_time_s"] == 1.0
    assert step["execution_timeline"]["intervals"][-1][
        "absolute_end_time_s"] == 1.03
    assert step["execution_timeline"]["host_events"] == ()
    assert step["execution_timeline"]["host_exposed_critical_path_s"] == 0.0
    body = {
        key: manifest[key]
        for key in manifest
        if key != "artifact_sha256"
    }
    serialized = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert manifest["artifact_sha256"] == hashlib.sha256(
        serialized).hexdigest()
    assert json.loads(json.dumps(manifest))["artifact_sha256"] == manifest[
        "artifact_sha256"]


def test_execution_artifact_serializes_predicted_host_events():
    result = _result()
    event = HostExecutionInterval(
        event_id="eager-submit:communication:single:rank-0",
        event_kind="eager_submit",
        pool_name="main",
        rank=0,
        call_id="communication",
        call_occurrence_index=-1,
        covered_call_occurrences=(("communication", -1),),
        device_predecessor_occurrences=(("vector", -1),),
        host_predecessor_event_id=None,
        runtime_graph_mode="NONE",
        graph_segment_id=None,
        start_time_s=0.0,
        end_time_s=0.001,
        exposed_time_s=0.001,
        overlapped_time_s=0.0,
    )
    step = result.steps[0]
    timeline = replace(
        step.execution_timeline,
        predicted_device_time_s=0.03,
        launch_cpu_time_s=0.001,
        launch_critical_path_s=0.001,
        host_device_overlap_s=0.001,
        eager_dispatch_count=1,
        launch_kinds=("eager",),
        host_intervals=(event,),
        runtime_graph_mode="NONE",
    )
    result = replace(
        result,
        steps=(replace(step, execution_timeline=timeline),),
    )

    host_events = _artifact(result)["steps"][0][
        "execution_timeline"]["host_events"]

    assert host_events == ({
        "event_id": "eager-submit:communication:single:rank-0",
        "event_kind": "eager_submit",
        "pool_name": "main",
        "rank": 0,
        "call_id": "communication",
        "call_occurrence_index": -1,
        "covered_call_occurrences": ({
            "call_id": "communication",
            "occurrence_index": -1,
        },),
        "device_predecessor_occurrences": ({
            "call_id": "vector",
            "occurrence_index": -1,
        },),
        "host_predecessor_event_id": None,
        "runtime_graph_mode": "NONE",
        "graph_segment_id": None,
        "clock_domain": "predicted_simulation_relative_s",
        "timing_source": "predicted",
        "exposure_scope": "lane_local_delay_before_call",
        "start_time_s": 0.0,
        "end_time_s": 0.001,
        "absolute_start_time_s": 1.0,
        "absolute_end_time_s": 1.001,
        "exposed_time_s": 0.001,
        "overlapped_time_s": 0.0,
    },)
    assert _artifact(result)["steps"][0]["execution_timeline"][
        "runtime_graph_mode"] == "NONE"


def test_execution_artifact_rejects_host_event_summary_mismatch():
    result = _result()
    step = result.steps[0]
    event = HostExecutionInterval(
        event_id="eager-submit:communication:single:rank-0",
        event_kind="eager_submit",
        pool_name="main",
        rank=0,
        call_id="communication",
        call_occurrence_index=-1,
        covered_call_occurrences=(("communication", -1),),
        device_predecessor_occurrences=(("vector", -1),),
        host_predecessor_event_id=None,
        runtime_graph_mode="NONE",
        graph_segment_id=None,
        start_time_s=0.0,
        end_time_s=0.001,
        exposed_time_s=0.0,
        overlapped_time_s=0.001,
    )
    inconsistent = replace(
        step.execution_timeline,
        launch_cpu_time_s=0.001,
        launch_critical_path_s=0.001,
        host_device_overlap_s=0.001,
        launch_kinds=("eager",),
        host_intervals=(event,),
        runtime_graph_mode="NONE",
    )

    with pytest.raises(ValueError, match="event count disagrees"):
        _artifact(replace(
            result,
            steps=(replace(step, execution_timeline=inconsistent),),
        ))


@pytest.mark.parametrize(
    "event_kind,runtime_graph_mode,graph_segment_id",
    (
        ("eager_submit", "FULL", None),
        ("graph_replay", "NONE", "segment-0"),
        ("graph_replay", "FULL", ""),
    ),
)
def test_host_event_rejects_inconsistent_runtime_descriptor(
        event_kind, runtime_graph_mode, graph_segment_id):
    with pytest.raises(ValueError):
        HostExecutionInterval(
            event_id="event-0",
            event_kind=event_kind,
            pool_name="main",
            rank=0,
            call_id="vector",
            call_occurrence_index=-1,
            covered_call_occurrences=(("vector", -1),),
            device_predecessor_occurrences=(),
            host_predecessor_event_id=None,
            runtime_graph_mode=runtime_graph_mode,
            graph_segment_id=graph_segment_id,
            start_time_s=0.0,
            end_time_s=0.001,
            exposed_time_s=0.001,
            overlapped_time_s=0.0,
        )


@pytest.mark.parametrize(
    "launch_policy,hardware_ids,expected",
    (
        ("analytical_launch", ("nvidia_a800",), _TIMING_STATUS),
        ("analytical_launch", ("ascend_910b",), "projected_by_midlayer"),
        ("analytical_launch", ("ascend_910b", "nvidia_a800"), _TIMING_STATUS),
        ("injected_launch", ("nvidia_a800",), "projected_by_midlayer"),
    ),
)
def test_performance_timing_status_is_derived_from_policy_and_hardware(
        launch_policy, hardware_ids, expected):
    assert performance_timing_status(launch_policy, hardware_ids) == expected


def test_development_protobridge_status_overrides_publishable_status():
    assert performance_timing_status(
        "injected_launch",
        ("metax_c500",),
        uses_development_protobridge=True,
    ) == "projected_with_protobridge_development_only"


def test_execution_artifact_rejects_unknown_performance_timing_status():
    with pytest.raises(ValueError, match="unsupported Serving performance"):
        execution_artifact_manifest(
            _result(),
            performance_timing_status="validated",
        )


def test_execution_artifact_accepts_large_clock_subtraction_noise():
    result = _result()
    entry = result.schedule_trace.entries[0]
    start_time_s = 1_000_000.0
    end_time_s = start_time_s + result.steps[0].execution_timeline.duration_s
    shifted_trace = ScheduleTrace((
        ScheduleTraceEntry(entry.backend_step, start_time_s, end_time_s),
    ))
    shifted_result = replace(result, schedule_trace=shifted_trace)

    manifest = _artifact(shifted_result)

    assert manifest["schedule_trace"]["entries"][0] == {
        "step_id": "main:vllm:0",
        "pool_name": "main",
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
    }


def test_execution_artifact_rejects_cross_owner_reserved_resource_overlap():
    step = {
        "step_id": "step-0",
        "execution_timeline": {
            "intervals": ({
                "call_id": "call-0",
                "reserved_resource_ids": ("communication:fabric:shared",),
                "absolute_start_time_s": 1.0,
                "absolute_end_time_s": 3.0,
            },),
        },
    }
    transfer = {
        "request_id": "request-0",
        "execution_timeline": {
            "intervals": ({
                "call_id": "transfer-0",
                "reserved_resource_ids": ("communication:fabric:shared",),
                "absolute_start_time_s": 2.0,
                "absolute_end_time_s": 4.0,
            },),
        },
    }

    with pytest.raises(ValueError, match="global intervals overlap"):
        _validate_global_reserved_resource_intervals((step,), (transfer,))


def test_execution_artifact_rejects_incomplete_prediction_coverage():
    result = _result()
    step = result.steps[0]
    broken = replace(
        step,
        predictions={"vector": step.predictions["vector"]},
    )

    with pytest.raises(ValueError, match="cover KernelPlan exactly"):
        _artifact(replace(result, steps=(broken,)))


def test_execution_artifact_rejects_non_json_workload_attributes():
    result = _result()
    step = result.steps[0]
    nodes = list(step.workload.nodes)
    nodes[0] = replace(nodes[0], attributes={"unsupported": object()})
    workload = WorkloadGraph(nodes, step.workload.edges)

    with pytest.raises(TypeError, match="non-JSON value"):
        _artifact(replace(
            result,
            steps=(replace(step, workload=workload),),
        ))


def test_execution_artifact_rejects_disconnected_workload_and_kernel_plan():
    result = _result()
    step = result.steps[0]
    unrelated = WorkloadGraph((
        WorkloadNode(
            "unrelated",
            WorkloadKind.ELEMENTWISE,
            "main",
            {"elements": 1},
            dtype="bf16",
            variant="elementwise",
            rank=0,
            attributes={
                "reads": 1,
                "writes": 1,
                "operation": "copy",
            },
        ),
    ))

    with pytest.raises(ValueError, match="unknown workload nodes"):
        _artifact(replace(
            result,
            steps=(replace(step, workload=unrelated),),
        ))


def test_execution_artifact_rejects_duplicate_simulation_steps():
    result = _result()

    with pytest.raises(ValueError, match="duplicate SimulationStep"):
        _artifact(replace(
            result,
            steps=(result.steps[0], result.steps[0]),
        ))


@pytest.mark.parametrize(
    "broken_interval,match",
    (
        (
            ExecutionInterval(
                "vector", "wrong-pool", 0, "compute", 0.0, 0.01,
                ("pool:main:rank:0:compute",)),
            "placement disagrees",
        ),
        (
            ExecutionInterval(
                "vector", "main", 0, "compute", 0.02, 0.01,
                ("pool:main:rank:0:compute",)),
            "end precedes start",
        ),
    ),
)
def test_execution_artifact_rejects_invalid_timeline_intervals(
        broken_interval, match):
    result = _result()
    step = result.steps[0]
    timeline = replace(
        step.execution_timeline,
        intervals=(broken_interval, step.execution_timeline.intervals[1]),
    )

    with pytest.raises(ValueError, match=match):
        _artifact(replace(
            result,
            steps=(replace(step, execution_timeline=timeline),),
        ))


def test_execution_artifact_validates_pd_transfer_end_to_end():
    manifest = _artifact(_result_with_pd_transfer())

    assert manifest["counts"]["simulation_transfers"] == 1
    transfer = manifest["transfers"][0]
    assert transfer["trace_transfer"]["request_ids"] == ("request-0",)
    assert transfer["kernel_plan"]["calls"][0]["source_node_ids"] == (
        "request-0:pd-kv-transfer:prefill-to-decode",)
    assert transfer["execution_timeline"]["origin_time_s"] == 1.03
    assert transfer["execution_timeline"]["intervals"][0][
        "absolute_end_time_s"] == pytest.approx(1.05)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("request_id", "wrong-request", "request disagrees"),
        ("source_engine_id", "wrong-source", "source disagrees"),
        ("destination_engine_id", "wrong-destination",
         "destination disagrees"),
    ),
)
def test_execution_artifact_rejects_pd_transfer_identity_mismatch(
        field, value, match):
    result = _result_with_pd_transfer()
    broken = replace(result.transfers[0], **{field: value})

    with pytest.raises(ValueError, match=match):
        _artifact(replace(result, transfers=(broken,)))


def test_execution_artifact_rejects_unmatched_kv_trace_transfer():
    result = _result_with_pd_transfer()

    with pytest.raises(ValueError, match="coverage disagree"):
        _artifact(replace(result, transfers=()))


def test_execution_artifact_rejects_duplicate_simulation_transfer():
    result = _result_with_pd_transfer()

    with pytest.raises(ValueError, match="duplicate SimulationTransfer"):
        _artifact(replace(
            result,
            transfers=(result.transfers[0], result.transfers[0]),
        ))


def test_execution_artifact_rejects_pd_transfer_without_matching_route():
    result = _result_with_pd_transfer()
    transfer = replace(result.transfers[0], request_id="orphan-request")
    trace_transfer = replace(
        result.schedule_trace.transfers[0],
        request_ids=("orphan-request",),
    )
    trace = ScheduleTrace(
        result.schedule_trace.entries,
        result.schedule_trace.route_decisions,
        (trace_transfer,),
    )

    with pytest.raises(ValueError, match="prefill_to_decode routes disagree"):
        _artifact(replace(
            result,
            schedule_trace=trace,
            transfers=(transfer,),
        ))


def test_execution_artifact_rejects_pd_transfer_without_source_completion():
    result = _result_with_pd_transfer()
    source_step = result.steps[0]
    schedule = replace(
        source_step.backend_step.schedule,
        finished_request_ids=(),
    )
    backend_step = replace(source_step.backend_step, schedule=schedule)
    broken_source_step = replace(source_step, backend_step=backend_step)
    entries = (
        replace(result.schedule_trace.entries[0], backend_step=backend_step),
        result.schedule_trace.entries[1],
    )
    trace = ScheduleTrace(
        entries,
        result.schedule_trace.route_decisions,
        result.schedule_trace.transfers,
    )

    with pytest.raises(ValueError, match="source Scheduler completion"):
        _artifact(replace(
            result,
            schedule_trace=trace,
            steps=(broken_source_step, result.steps[1]),
        ))
