from dataclasses import replace
from types import SimpleNamespace

import pytest

from evaluation.serving.simulation.execution_artifact import (
    PROJECTED_PERFORMANCE_TIMING_STATUS,
    StreamingExecutionArtifactAccumulator,
    execution_artifact_manifest,
)
from evaluation.serving.metrics.system_resource_analysis import (
    system_resource_manifest,
)
from evaluation.serving.simulation.simulation_runner import (
    _ConformanceBackend,
    _host_timing_manifest,
)
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import KernelKind, KernelPrediction, KernelSpec
from src.kernel.launch_config import LaunchConfig
from src.serving.deployment.af_disagg import build_af_disagg_plan
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    PipelineP2PConfig,
    RouteConfig,
    WorkerPoolConfig,
)
from src.serving.execution import ServingExecutionContext
from src.serving.host_timing import HostLaunchProfile
from src.serving.kernel_lowering.execution_timeline import (
    build_execution_timeline,
)
from src.serving.kernel_lowering.kernel_plan import (
    KernelBlockRepeat,
    KernelCall,
    KernelPlan,
)
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduleStep,
    ScheduledRequest,
)
from src.serving.schedule.schedule_trace import ScheduleTrace, ScheduleTraceEntry
from src.serving.simulation import Simulation, SimulationConfig
from src.serving.simulation import (
    SimulationStep,
    StreamingSimulationResult,
    _with_in_step_transfer_trace,
)
from src.serving.workload.build_workload import (
    BlockExecutionSignature,
    BlockConstructionMemo,
    TargetBlockExecutionSignature,
    build_workload,
)
from src.serving.workload.model_config import load_model_config
from src.serving.workload.expert_frequency import load_expert_frequency_profile
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs


def _assert_step_logically_equivalent(reference, accelerated):
    assert accelerated.execution_timeline.duration_s == (
        reference.execution_timeline.duration_s)
    assert accelerated.workload.logical_node_count == len(
        reference.workload.nodes)
    assert accelerated.workload.logical_edge_count == len(
        reference.workload.edges)
    assert accelerated.kernel_plan.logical_call_count == len(
        reference.kernel_plan.calls)
    assert sum(
        count for _call_id, count
        in accelerated.execution_timeline.call_occurrence_counts
    ) == len(reference.execution_timeline.intervals)
    assert accelerated.execution_timeline.pool_activity == (
        reference.execution_timeline.pool_activity)
    assert accelerated.execution_timeline.reserved_resource_activity == (
        reference.execution_timeline.reserved_resource_activity)
    assert accelerated.execution_timeline.critical_path_contributions == (
        reference.execution_timeline.critical_path_contributions)
    assert accelerated.execution_timeline.critical_call_count == (
        reference.execution_timeline.critical_call_count)
    assert len(accelerated.workload.nodes) < len(reference.workload.nodes)
    assert len(accelerated.kernel_plan.calls) < len(reference.kernel_plan.calls)
    assert len(accelerated.predictions) < len(reference.predictions)
    assert len(accelerated.execution_timeline.intervals) < len(
        reference.execution_timeline.intervals)


def _assert_result_logically_equivalent(reference, accelerated):
    assert accelerated.schedule_trace == reference.schedule_trace
    assert accelerated.metrics == reference.metrics
    assert len(accelerated.steps) == len(reference.steps)
    for reference_step, accelerated_step in zip(
            reference.steps, accelerated.steps, strict=True):
        _assert_step_logically_equivalent(reference_step, accelerated_step)
    reference_artifact = execution_artifact_manifest(
        reference,
        performance_timing_status=PROJECTED_PERFORMANCE_TIMING_STATUS,
    )
    accelerated_artifact = execution_artifact_manifest(
        accelerated,
        performance_timing_status=PROJECTED_PERFORMANCE_TIMING_STATUS,
    )
    assert reference_artifact["schema_version"] == 5
    assert accelerated_artifact["schema_version"] == 8
    assert accelerated_artifact["counts"] == reference_artifact["counts"]


def test_in_step_transfer_projection_preserves_pipeline_queue_timing():
    backend_step = BackendStep(
        ScheduleStep(
            "main:vllm:0",
            "main",
            (ScheduledRequest("request-0", "prefill", 1, 0),),
        ),
        generated_tokens={"request-0": 0},
    )
    entry = ScheduleTraceEntry(
        backend_step,
        1.0,
        2.0,
        timing_mode="device_plus_launch_plus_scheduler",
        scheduler_plan_cpu_s=0.1,
        scheduler_update_cpu_s=0.1,
        execution_wall_s=0.5,
        device_queue_wait_s=0.2,
        completion_queue_wait_s=0.1,
    )
    simulation_step = SimulationStep(
        backend_step,
        SimpleNamespace(block_repeats=(), nodes=()),
        SimpleNamespace(calls=()),
        {},
        SimpleNamespace(intervals=(), block_repeats=()),
    )

    projected = _with_in_step_transfer_trace(
        ScheduleTrace((entry,)), (simulation_step,))

    assert projected.entries == (entry,)


def _simulation(
    case: str,
    hardware_id: str,
    execution_mode: str,
    *,
    num_layers: int = 4,
) -> Simulation:
    if case == "dense_tp2":
        model = load_model_config(
            "tests/serving/fixtures/qwen3_tp2_scheduler")
        model = replace(
            model,
            num_hidden_layers=num_layers,
            num_dense_layers=num_layers,
            head_dim=128,
        )
        routing = ExpertRoutingInput.not_applicable()
        enable_expert_parallel = False
    elif case in ("moe_tp2_ep2", "moe_profile_tp2_ep2"):
        model = load_model_config(
            "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
        model = replace(
            model,
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=32,
            num_key_value_heads=4,
            num_hidden_layers=num_layers,
            head_dim=128,
            vocab_size=151936,
            max_position_embeddings=40960,
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
        )
        if case == "moe_tp2_ep2":
            routing = ExpertRoutingInput.seeded_assumption(
                "execution-equivalence-v1", 17)
        else:
            profile = load_expert_frequency_profile(
                "serving_dataset/moe_expert_freq/manifest.json",
                "Qwen3-30B-A3B",
                expected_model_id="Qwen3-30B-A3B",
                expected_moe_layers=48,
                expected_num_experts=128,
            )
            routing = ExpertRoutingInput.profile_observation(profile)
        enable_expert_parallel = True
    else:
        raise AssertionError(f"unknown test case: {case!r}")
    collectives = (
        CollectiveConfig(
            "all_reduce",
            "ring",
            "nvlink",
            "ring",
            "main-tensor-fabric",
            "analytical_roofline_v1",
        ),
        CollectiveConfig(
            "all_gather",
            "ring",
            "nvlink",
            "ring",
            "main-tensor-fabric",
            "analytical_roofline_v1",
        ),
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=enable_expert_parallel,
                ),
                ranks=(0, 1),
                collectives=collectives,
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    hardware.interconnect["nccl_alpha_ns"] = 100.0
    hardware.interconnect["nvlink_signal_latency_ns"] = 100.0
    return Simulation(
        SimulationConfig(
            model,
            deployment,
            RequestStream((RequestSpec(0.0, "request-0", 4, 2),)),
            "analytical_launch",
            "midlayer",
            RuntimeInputs(routing, "paged_varlen", "uniform"),
            execution_mode=execution_mode,
        ),
        _ConformanceBackend(BackendConfig("main", "colocated", 128, 4, 128, scheduler_pipeline_depth=1)),
        {hardware_id: hardware},
    )


@pytest.mark.parametrize("num_layers", (2, 8, 36))
def test_dense_block_reuse_keeps_materialized_execution_constant(num_layers):
    accelerated = _simulation(
        "dense_tp2",
        "nvidia_a800",
        "accelerated",
        num_layers=num_layers,
    ).run()
    step = accelerated.steps[0]
    baseline = _simulation(
        "dense_tp2",
        "nvidia_a800",
        "accelerated",
        num_layers=2,
    ).run().steps[0]

    assert len(step.workload.nodes) == len(baseline.workload.nodes)
    assert len(step.kernel_plan.calls) == len(baseline.kernel_plan.calls)
    assert len(step.predictions) == len(baseline.predictions)
    assert len(step.execution_timeline.intervals) == len(
        baseline.execution_timeline.intervals)
    if num_layers == 2:
        assert step.workload.block_repeats == ()
        assert step.kernel_plan.block_repeats == ()
    else:
        assert len(step.workload.block_repeats) == 1
        assert step.workload.block_repeats[0].repeat_count == num_layers - 1
        assert step.kernel_plan.block_repeats[0].repeat_count == num_layers - 1
        assert step.execution_timeline.block_repeats[0].repeat_count == (
            num_layers - 1)
    assert step.workload.logical_node_count == (
        len(baseline.workload.nodes)
        + (num_layers - 2) * len(
            step.workload.block_repeats[0].node_ids)
        if num_layers > 2 else len(baseline.workload.nodes)
    )


def test_repeat_successors_wait_for_the_final_logical_occurrence():
    step = _simulation(
        "dense_tp2", "nvidia_a800", "accelerated", num_layers=8,
    ).run().steps[0]
    repeat = step.kernel_plan.block_repeats[0]
    repeated_call_ids = set(repeat.call_ids)
    timeline_repeat = step.execution_timeline.block_repeats[0]
    occurrence_intervals = dict(timeline_repeat.call_intervals)
    representative_intervals = {
        interval.call_id: interval
        for interval in step.execution_timeline.intervals
    }
    successor_edges = tuple(
        (call.call_id, dependency)
        for call in step.kernel_plan.calls
        if call.call_id not in repeated_call_ids
        for dependency in call.dependencies
        if dependency in repeated_call_ids
    )

    assert successor_edges
    for successor_id, dependency_id in successor_edges:
        assert representative_intervals[successor_id].start_time_s >= (
            occurrence_intervals[dependency_id][-1][1])


def test_compact_repeat_preserves_global_kahn_order_across_resources():
    spec = KernelSpec(
        KernelKind.VECTOR,
        "nvidia_a800",
        {"element_count": 1},
        dtype="bf16",
        variant="elementwise",
        launch=LaunchConfig(
            "repeat-order-fixture",
            (1, 1, 1),
            (1, 1, 1),
            execution_variant="elementwise",
        ),
        attributes={"reads": 1, "writes": 1, "operation": "copy"},
    )

    def call(call_id, pool_name, dependencies=()):
        return KernelCall(
            call_id,
            spec,
            pool_name,
            0,
            "compute",
            (f"node:{call_id}",),
            dependencies,
        )

    compact_plan = KernelPlan(
        (
            call("B", "shared"),
            call("seed", "seed"),
            call("A", "independent", ("seed",)),
            call("C", "shared", ("B", "A")),
        ),
        (KernelBlockRepeat(
            "repeat",
            ("B", "A", "C"),
            3,
            (("A", ("C",)),),
        ),),
    )
    expanded_plan = KernelPlan((
        call("B0", "shared"),
        call("B1", "shared"),
        call("B2", "shared"),
        call("seed", "seed"),
        call("A0", "independent", ("seed",)),
        call("C0", "shared", ("B0", "A0")),
        call("A1", "independent", ("C0",)),
        call("C1", "shared", ("B1", "A1")),
        call("A2", "independent", ("C1",)),
        call("C2", "shared", ("B2", "A2")),
    ))

    def prediction(duration_s):
        return KernelPrediction(
            duration_s,
            {"fixture": duration_s},
            0.0,
            "fixture",
            {"fixture": duration_s},
        )

    compact_predictions = {
        "B": prediction(10.0),
        "seed": prediction(1.0),
        "A": prediction(10.0),
        "C": prediction(1.0),
    }
    expanded_predictions = {
        call_id: prediction(
            1.0 if call_id == "seed" or call_id.startswith("C") else 10.0)
        for call_id in (
            "B0", "B1", "B2", "seed", "A0", "C0", "A1", "C1",
            "A2", "C2",
        )
    }

    reference = build_execution_timeline(
        expanded_plan, expanded_predictions)
    compact = build_execution_timeline(compact_plan, compact_predictions)
    summary = build_execution_timeline(
        compact_plan,
        compact_predictions,
        materialize_timeline_details=False,
    )
    reference_intervals = {
        interval.call_id: (interval.start_time_s, interval.end_time_s)
        for interval in reference.intervals
    }
    compact_intervals = dict(compact.block_repeats[0].call_intervals)

    assert reference.duration_s == compact.duration_s == summary.duration_s == 53.0
    assert compact_intervals == {
        "B": tuple(reference_intervals[f"B{index}"] for index in range(3)),
        "A": tuple(reference_intervals[f"A{index}"] for index in range(3)),
        "C": tuple(reference_intervals[f"C{index}"] for index in range(3)),
    }
    assert compact_intervals == {
        "B": ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0)),
        "A": ((1.0, 11.0), (31.0, 41.0), (42.0, 52.0)),
        "C": ((30.0, 31.0), (41.0, 42.0), (52.0, 53.0)),
    }
    assert summary.intervals == ()
    assert summary.block_repeats[0].call_intervals == ()
    assert summary.block_repeats[0].call_interval_sha256


def test_accelerated_serving_matches_reference_smoke():
    reference = _simulation("dense_tp2", "nvidia_a800", "reference").run()
    accelerated = _simulation("dense_tp2", "nvidia_a800", "accelerated").run()

    _assert_result_logically_equivalent(reference, accelerated)
    assert accelerated.execution_cache_stats["execution_mode"] == "accelerated"
    assert accelerated.execution_cache_stats["kernel_prediction"]["hits"] > 0
