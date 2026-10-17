"""Canonical structured formula tests."""

import dataclasses
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.kernel.dataset_storage import load_kernel_dataset_document

import src.kernel.formula as formula_module
from src.kernel.formula import (
    PerformanceFormula,
    _split_fuse_mm2_source_order_time,
)
from src.kernel.hardware_spec import HWSpec
from src.kernel.model_contract import (
    AIV_COMPLETE_TASK_FLOOR_ONLY,
    AIV_SOURCE_ENTRY_FILL,
    AIV_SOURCE_PIPELINE_LAUNCH_ONLY,
    AIV_WORKER_START_FILL,
    CANN_AIV_TILING_MATERIALIZATION_PROLOGUE,
    AnalyticalDependencyPath,
    ComputeExactRequestDemand,
    ComputeServiceDemand,
    FiaAivMemoryStageDemand,
    FiaBlockReduceSumDescriptorDemand,
    FiaComputeStageDemand,
    FiaCrossDomainAicGroupDemand,
    FiaCrossDomainTaskDemand,
    FiaCrossDomainWorkPlan,
    FiaSplitFuseRequestRunDemand,
    FiaSplitFuseWorkPlan,
    FiaFixpipeDescriptorDemand,
    FiaFixpipeStageDemand,
    FiaGqaSourceOrderDemand,
    FiaGqaSourcePanelDemand,
    FiaGqaSourceRoundDemand,
    FiaMte1LoadPairDemand,
    FiaMte1StageDemand,
    FiaMte2Nd2NzDescriptorDemand,
    FiaMte2StageDemand,
    FiaTilingDataPreludeDemand,
    MemoryAccessClassDemand,
    MemoryAccessClassSignature,
    MemoryExactRequestDemand,
    PhaseMemoryDemand,
    PipelineStageDemand,
    PipelineTaskDemand,
    PipelineWorkPlan,
)
from src.kernel.predictors.common import (
    bind_memory_access_class_template,
    bind_pipeline_stage_template,
    bind_typed_compute_pipeline_stage_template,
    PredictionInput,
    append_npu_write_path,
    attach_pipeline_work_plan,
    attach_structured_contract,
    attach_uniform_pipeline,
    gpu_memory_source,
    replace_structured_demands,
    split_global_memory_paths,
)


HW = HWSpec("configs/hardware/nvidia_a800.yaml")
FORMULA = PerformanceFormula(HW)
NPU_HW = HWSpec("configs/hardware/ascend_910b.yaml")
NPU_FORMULA = PerformanceFormula(NPU_HW)
C500_HW = HWSpec("configs/hardware/metax_c500.yaml")


def _fia_mte2_stage(worker):
    return FiaMte2StageDemand(
        worker,
        (FiaMte2Nd2NzDescriptorDemand(4, 64, 2, 1),),
    )


def _fia_mte1_stage(worker):
    return FiaMte1StageDemand(worker, 0, 0)


def _fia_fixpipe_stage(worker, count=0):
    descriptors = () if count == 0 else (
        FiaFixpipeDescriptorDemand(4, 128, 128, count),)
    return FiaFixpipeStageDemand(worker, descriptors)


def test_fia_zero_work_program_reduces_to_measured_ring_span():
    groups = []
    for aic_worker in range(1):
        tasks = []
        for ordinal in range(1):
            tensor = FiaComputeStageDemand(
                "compute_tensor", aic_worker, 0.0, 0.0)
            vector = FiaComputeStageDemand(
                "compute_vector", 2 * aic_worker, 0.0, 0.0)
            tasks.append(FiaCrossDomainTaskDemand(
                ordinal=ordinal,
                mm1_mte2=_fia_mte2_stage(aic_worker),
                mm1_mte1=_fia_mte1_stage(aic_worker),
                mm1=tensor,
                mm1_fixpipe=_fia_fixpipe_stage(aic_worker),
                vec1=vector,
                mm2_mte2=_fia_mte2_stage(aic_worker),
                mm2_mte1=_fia_mte1_stage(aic_worker),
                mm2=tensor,
                mm2_fixpipe=_fia_fixpipe_stage(aic_worker),
                vec2=vector,
            ))
        groups.append(FiaCrossDomainAicGroupDemand(
            aic_worker=aic_worker,
            aiv_workers=(2 * aic_worker, 2 * aic_worker + 1),
            tasks=tuple(tasks),
        ))
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5", tuple(groups), ())

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})

    sync_all = NPU_HW.fia_aiv_sync_all_time_ns(2) * 1e-9
    expected = (
        sync_all
        + NPU_HW.fia_cross_domain_three_handoff_time_ns((1,)) * 1e-9)
    assert diagnostics.execution_time_s == pytest.approx(expected)
    assert diagnostics.critical_path == (
        "fia_output_init_syncall", "fia_cross_domain_program_span")
    assert diagnostics.stage_service_s[
        "fia_output_init_syncall"] == pytest.approx(sync_all)
    assert diagnostics.stage_service_s["compute_tensor"] == 0.0
    assert diagnostics.stage_service_s["compute_vector"] == 0.0
    assert diagnostics.stage_service_s["aic_mte2_nd2nz"] > 0.0

    nonzero_task = dataclasses.replace(
        plan.groups[0].tasks[0],
        mm1=FiaComputeStageDemand("compute_tensor", 0, 1.0, 1.0),
        vec1=FiaComputeStageDemand(
            "compute_vector", 0, 1.0, 1.0,
            (ComputeServiceDemand("generic", 1.0),)),
        mm2=FiaComputeStageDemand("compute_tensor", 0, 1.0, 1.0),
        vec2=FiaComputeStageDemand(
            "compute_vector", 0, 1.0, 1.0,
            (ComputeServiceDemand("generic", 1.0),)),
    )
    nonzero_plan = FiaCrossDomainWorkPlan(
        plan.mechanism_id,
        (dataclasses.replace(plan.groups[0], tasks=(nonzero_task,)),),
        (),
    )
    base = _make_input(
        contract_hardware=NPU_HW,
        active_units=1,
        grid_size=1,
        launch_task_kind="mixed",
        include_dispatch_overhead=True,
        demand_per_resource={
            "compute_tensor": 2.0,
            "compute_vector": 2.0,
        },
    )
    typed_compute_demands = []
    for demand in base.compute_demands:
        if demand.resource_role == "compute_tensor":
            typed_compute_demands.append(demand)
        elif demand.resource_role == "compute_vector":
            typed_compute_demands.append(dataclasses.replace(
                demand,
                service_demands=(ComputeServiceDemand(
                    "generic", 2.0,
                    operations_by_worker=(2.0, 0.0),
                    serial_invocations_by_worker=(0, 0)),),
            ))
        else:
            raise AssertionError(demand.resource_role)
    prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
        base,
        compute_demands=tuple(typed_compute_demands),
        pipeline_plan=nonzero_plan,
        support_level="partial",
    ))
    outer_s = NPU_HW.fia_mixed_task_outer_envelope_time_ns(2) * 1e-9
    assert prediction.t_launch_s == pytest.approx(outer_s)
    assert prediction.t_total_s == pytest.approx(
        outer_s + prediction.diagnostics.pipeline.execution_time_s)


def test_fia_tiling_preludes_gate_their_own_execution_domains():
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=_fia_mte2_stage(0),
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector,
        mm2_mte2=_fia_mte2_stage(0),
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector,
    )
    baseline_plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )
    prelude = FiaTilingDataPreludeDemand(
        "fia_mixed_tiling_data_materialization_v1", 1560, 1568, 1)
    plan = dataclasses.replace(
        baseline_plan, tiling_data_prelude=prelude)

    baseline = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        baseline_plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    aic_ns, aiv_ns = NPU_HW.fia_tiling_data_materialization_times_ns(
        1, 1560, 1568)

    assert diagnostics.stage_service_s[
        "fia_aic_tiling_data_materialization"] == pytest.approx(
            aic_ns * 1e-9)
    assert diagnostics.stage_service_s[
        "fia_aiv_tiling_data_materialization"] == pytest.approx(
            aiv_ns * 1e-9)
    assert diagnostics.execution_time_s - baseline.execution_time_s == (
        pytest.approx(aiv_ns * 1e-9))
    task_diagnostic, = diagnostics.tasks
    assert task_diagnostic.stage_start_s["mm1_fixpipe"] == pytest.approx(
        aic_ns * 1e-9)
    assert diagnostics.critical_path[0] == (
        "fia_aiv_tiling_data_materialization")


def test_fia_fixpipe_completion_precedes_vector_handoff():
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=_fia_mte2_stage(0),
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0, count=2),
        vec1=vector,
        mm2_mte2=_fia_mte2_stage(0),
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0, count=1),
        vec2=vector,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    one_call_s = NPU_HW.aic_fixpipe_full_tile_time_ns(
        4, 128, 128) * 1e-9

    assert diagnostics.stage_service_s[
        "aic_fixpipe_l0c_to_gm"] == pytest.approx(3 * one_call_s)
    task_diagnostic, = diagnostics.tasks
    assert task_diagnostic.stage_start_s["vec1"] == pytest.approx(
        task_diagnostic.stage_end_s["mm1_fixpipe"])
    assert task_diagnostic.stage_start_s["vec2"] == pytest.approx(
        task_diagnostic.stage_end_s["mm2_fixpipe"])


def test_fia_aiv_memory_stages_are_ordered_on_the_task_timeline():
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=_fia_mte2_stage(0),
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector,
        mm2_mte2=_fia_mte2_stage(0),
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector,
        vec1_memory=FiaAivMemoryStageDemand(0, 1_000_000, 2_000_000),
        vec2_memory=FiaAivMemoryStageDemand(0, 3_000_000, 4_000_000),
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )
    baseline = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        dataclasses.replace(
            plan,
            groups=(dataclasses.replace(
                plan.groups[0],
                tasks=(dataclasses.replace(
                    task, vec1_memory=None, vec2_memory=None),),
            ),),
        ),
        active_units=1,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )
    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan,
        active_units=1,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )
    mte2 = NPU_HW.resource_by_role("mte2")
    mte3 = NPU_HW.resource_by_role("mte3")
    mte2_per_aiv = mte2.effective_capacity(
        mte2.total_units,
        sm_freq_hz=NPU_HW.nominal_sm_freq_hz,
        dram_freq_hz=NPU_HW.nominal_dram_freq_hz,
    ) / mte2.total_units
    mte3_per_aiv = mte3.effective_capacity(
        mte3.total_units,
        sm_freq_hz=NPU_HW.nominal_sm_freq_hz,
        dram_freq_hz=NPU_HW.nominal_dram_freq_hz,
    ) / mte3.total_units
    expected_mte2_s = 4_000_000 / mte2_per_aiv
    expected_mte3_s = 6_000_000 / mte3_per_aiv

    assert diagnostics.stage_service_s["aiv_mte2_gm_to_ub"] == (
        pytest.approx(expected_mte2_s))
    assert diagnostics.stage_service_s["aiv_mte3_ub_to_gm"] == (
        pytest.approx(expected_mte3_s))
    task_diagnostic, = diagnostics.tasks
    baseline_task, = baseline.tasks
    vec1_delta = (
        task_diagnostic.stage_end_s["vec1"]
        - task_diagnostic.stage_start_s["vec1"]
        - baseline_task.stage_end_s["vec1"]
        + baseline_task.stage_start_s["vec1"])
    vec2_delta = (
        task_diagnostic.stage_end_s["vec2"]
        - task_diagnostic.stage_start_s["vec2"]
        - baseline_task.stage_end_s["vec2"]
        + baseline_task.stage_start_s["vec2"])
    assert vec1_delta == pytest.approx(
        1_000_000 / mte2_per_aiv + 2_000_000 / mte3_per_aiv)
    assert vec2_delta == pytest.approx(
        3_000_000 / mte2_per_aiv + 4_000_000 / mte3_per_aiv)


def test_fia_aiv_transfer_completion_is_a_non_additive_lower_bound(
        monkeypatch):
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=_fia_mte2_stage(0),
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector,
        mm2_mte2=_fia_mte2_stage(0),
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector,
        vec1_memory=FiaAivMemoryStageDemand(
            0, 1, 1, "vec1_m4_s512_ordered_chain"),
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )
    calls = []

    def completion_time_ns(mode, active_groups):
        calls.append((mode, active_groups))
        return 1_000.0

    monkeypatch.setattr(
        NPU_HW, "fia_aiv_workspace_transfer_time_ns", completion_time_ns)
    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan,
        active_units=1,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )

    task_diagnostic, = diagnostics.tasks
    assert calls == [("vec1_m4_s512_ordered_chain", 1)]
    queue_cycle_s = NPU_HW.resource_by_role(
        "scalar_aiv").exact_request_service_time_s(
            "aiv_queue_control_cycle", 1)
    assert (
        task_diagnostic.stage_end_s["vec1"]
        - task_diagnostic.stage_start_s["vec1"]
    ) == pytest.approx(1_000e-9 + 2 * queue_cycle_s)


def test_fia_mte1_load_pair_completion_is_a_non_additive_lower_bound():
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    mm1_mte1 = FiaMte1StageDemand(
        0, 2_048, 16_384, FiaMte1LoadPairDemand("mm1", 64, 1))
    mm2_mte1 = FiaMte1StageDemand(
        0, 4_096, 16_384, FiaMte1LoadPairDemand("mm2", 64, 1))
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=_fia_mte2_stage(0),
        mm1_mte1=mm1_mte1,
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector,
        mm2_mte2=_fia_mte2_stage(0),
        mm2_mte1=mm2_mte1,
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    expected_ns = (
        NPU_HW.aic_mte1_load_pair_time_ns("mm1", 64, 1)
        + NPU_HW.aic_mte1_load_pair_time_ns("mm2", 64, 1))

    assert diagnostics.stage_service_s["aic_mte1_l1_to_l0"] == (
        pytest.approx(expected_ns * 1e-9))


def test_fia_gqa_repeated_mte2_descriptors_each_use_serial_completion(
        monkeypatch):
    stream_calls = []
    original_stream_time = NPU_HW.aic_mte2_nd2nz_stream_time_ns

    def record_stream_time(descriptor_bytes, active_aic_groups):
        stream_calls.append((descriptor_bytes, active_aic_groups))
        return original_stream_time(descriptor_bytes, active_aic_groups)

    monkeypatch.setattr(
        NPU_HW, "aic_mte2_nd2nz_stream_time_ns", record_stream_time)
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    repeated = FiaMte2StageDemand(
        0, (FiaMte2Nd2NzDescriptorDemand(4, 256, 2, 2),))
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=repeated,
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector,
        mm2_mte2=repeated,
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    serial_ns = NPU_HW.aic_mte2_nd2nz_serial_time_ns(2048, 1)

    assert stream_calls == []
    assert diagnostics.stage_service_s["aic_mte2_nd2nz"] == pytest.approx(
        4 * serial_ns * 1e-9)


def test_fia_gqa_three_task_cache_overlaps_only_steady_mte2():
    tensor = FiaComputeStageDemand(
        "compute_tensor", 0, 1_000_000.0, 1_000_000.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)

    def resolve(repeat_count, source_stride):
        mte2 = FiaMte2StageDemand(0, (
            FiaMte2Nd2NzDescriptorDemand(
                4, 64, 2, 1,
                source_row_stride_elements=source_stride),
        ))
        task = FiaCrossDomainTaskDemand(
            ordinal=0,
            mm1_mte2=mte2,
            mm1_mte1=_fia_mte1_stage(0),
            mm1=tensor,
            mm1_fixpipe=_fia_fixpipe_stage(0, count=1),
            vec1=vector,
            mm2_mte2=mte2,
            mm2_mte1=_fia_mte1_stage(0),
            mm2=tensor,
            mm2_fixpipe=_fia_fixpipe_stage(0, count=1),
            vec2=vector,
            repeat_count=repeat_count,
        )
        plan = FiaCrossDomainWorkPlan(
            "cann_fia_1_2_three_handoff_v5",
            (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
            (),
        )
        return NPU_FORMULA._resolve_fia_cross_domain_work_plan(
            plan, active_units=1, launch_task_kind="mixed",
            resource_time_overrides_ns={})

    two_task_serial = resolve(2, None)
    two_task_buffered = resolve(2, 64)
    assert two_task_buffered.worker_makespan_s["aic:0"] == pytest.approx(
        two_task_serial.worker_makespan_s["aic:0"])

    three_task_serial = resolve(3, None)
    three_task_buffered = resolve(3, 64)
    assert three_task_buffered.worker_makespan_s["aic:0"] < (
        three_task_serial.worker_makespan_s["aic:0"])
    for stage in (
            "aic_mte2_nd2nz", "aic_mte1_l1_to_l0", "compute_tensor",
            "aic_fixpipe_l0c_to_gm"):
        assert three_task_buffered.stage_service_s[stage] == pytest.approx(
            three_task_serial.stage_service_s[stage])


def test_fia_small_probability_descriptor_uses_minimum_transaction(
        monkeypatch):
    serial_calls = []
    original_serial_time = NPU_HW.aic_mte2_nd2nz_serial_time_ns

    def record_serial_time(descriptor_bytes, active_aic_groups):
        serial_calls.append((descriptor_bytes, active_aic_groups))
        return original_serial_time(descriptor_bytes, active_aic_groups)

    monkeypatch.setattr(
        NPU_HW, "aic_mte2_nd2nz_serial_time_ns", record_serial_time)
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    small = FiaMte2StageDemand(
        0, (FiaMte2Nd2NzDescriptorDemand(
            4, 2, 2, 2, source_row_stride_elements=544),))
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=small,
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0, count=1),
        vec1=vector,
        mm2_mte2=small,
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0, count=1),
        vec2=vector,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )

    NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})

    minimum_bytes = NPU_HW.aic_mte2_nd2nz_service["descriptor_bytes"][0]
    assert serial_calls == [(minimum_bytes, 1)]


def test_fia_gqa_unknown_stride_uses_bounded_cold_stage_stream(monkeypatch):
    cold_stream_calls = []

    def cold_stream_time(aggregate_dram_bytes, active_aic_groups):
        cold_stream_calls.append((aggregate_dram_bytes, active_aic_groups))
        return 100_000.0

    monkeypatch.setattr(
        NPU_HW, "aic_mte2_nd2nz_cold_stream_time_ns", cold_stream_time)
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    mte2 = FiaMte2StageDemand(0, (
        FiaMte2Nd2NzDescriptorDemand(
            4, 128, 2, 1, source_row_stride_elements=128),
        FiaMte2Nd2NzDescriptorDemand(128, 128, 2, 128),
    ))
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=mte2,
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector,
        mm2_mte2=mte2,
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )
    memory = SimpleNamespace(
        path_name="mte2_read",
        nodes=(
            SimpleNamespace(resource_role="l2_cache", bytes=1_000.0),
            SimpleNamespace(resource_role="dram", bytes=500.0),
        ),
    )

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={}, memory_diagnostics=(memory,))

    expected_dram_bytes = mte2.total_bytes * 0.5
    assert cold_stream_calls == [
        (expected_dram_bytes, 1),
        (expected_dram_bytes, 1),
    ]
    assert diagnostics.stage_service_s["aic_mte2_nd2nz"] == pytest.approx(
        200_000.0e-9)


@pytest.mark.parametrize("rows,columns,count", (
    (129, 128, 1),
    (128, 64, 1),
    (128, 128, 161),
))
def test_fia_gqa_cold_stage_stream_rejects_unmeasured_programs(
        monkeypatch, rows, columns, count):
    cold_stream_calls = []

    def cold_stream_time(aggregate_dram_bytes, active_aic_groups):
        cold_stream_calls.append((aggregate_dram_bytes, active_aic_groups))
        return 100_000.0

    monkeypatch.setattr(
        NPU_HW, "aic_mte2_nd2nz_cold_stream_time_ns", cold_stream_time)
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    mte2 = FiaMte2StageDemand(0, (
        FiaMte2Nd2NzDescriptorDemand(
            4, 128, 2, 1, source_row_stride_elements=128),
        FiaMte2Nd2NzDescriptorDemand(rows, columns, 2, count),
    ))
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=mte2,
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector,
        mm2_mte2=mte2,
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )
    memory = SimpleNamespace(
        path_name="mte2_read",
        nodes=(
            SimpleNamespace(resource_role="l2_cache", bytes=1_000.0),
            SimpleNamespace(resource_role="dram", bytes=500.0),
        ),
    )

    NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={}, memory_diagnostics=(memory,))

    assert cold_stream_calls == []


@pytest.mark.parametrize(
    ("rows", "columns", "source_stride", "active_groups",
     "descriptor_count", "repeat_count", "aggregate_descriptor_count"),
    (
        (256, 64, 512, 16, 2, 1, 2),
        (256, 64, 512, 16, 4, 1, 4),
        (256, 128, 1024, 8, 16, 1, 16),
        (256, 128, 1024, 8, 32, 1, 32),
        (256, 128, 1024, 16, 8, 1, 8),
        (256, 128, 1024, 16, 16, 1, 16),
        (256, 128, 1024, 16, 32, 1, 32),
        (256, 128, 1024, 22, 4, 1, 4),
        (256, 128, 1024, 22, 8, 1, 8),
        (256, 128, 1024, 22, 16, 1, 16),
        (128, 128, 1024, 22, 8, 1, 8),
        (256, 128, 1024, 24, 2, 1, 2),
        (256, 128, 1024, 24, 2, 10, 20),
        (256, 128, 1024, 24, 2, 11, 22),
    ),
)
def test_fia_gqa_uses_exact_cold_finite_drain(
        monkeypatch, rows, columns, source_stride, active_groups,
        descriptor_count, repeat_count, aggregate_descriptor_count):
    finite_drain_calls = []
    original_finite_drain = (
        NPU_HW.aic_mte2_fia_gqa_finite_drain_time_ns)

    def record_finite_drain(
            rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count):
        finite_drain_calls.append((
            rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count))
        return original_finite_drain(
            rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count)

    monkeypatch.setattr(
        NPU_HW, "aic_mte2_fia_gqa_finite_drain_time_ns",
        record_finite_drain)
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    groups = []
    for worker in range(active_groups):
        mte2 = FiaMte2StageDemand(worker, (
            FiaMte2Nd2NzDescriptorDemand(
                4, columns, 2, 1,
                source_row_stride_elements=columns),
            FiaMte2Nd2NzDescriptorDemand(
                rows, columns, 2, descriptor_count,
                source_row_stride_elements=source_stride),
        ))
        task = FiaCrossDomainTaskDemand(
            ordinal=0,
            mm1_mte2=mte2,
            mm1_mte1=_fia_mte1_stage(worker),
            mm1=dataclasses.replace(tensor, physical_worker=worker),
            mm1_fixpipe=_fia_fixpipe_stage(worker, count=2),
            vec1=dataclasses.replace(vector, physical_worker=2 * worker),
            mm2_mte2=mte2,
            mm2_mte1=_fia_mte1_stage(worker),
            mm2=dataclasses.replace(tensor, physical_worker=worker),
            mm2_fixpipe=_fia_fixpipe_stage(worker, count=2),
            vec2=dataclasses.replace(vector, physical_worker=2 * worker),
            repeat_count=repeat_count,
        )
        groups.append(FiaCrossDomainAicGroupDemand(
            worker, (2 * worker, 2 * worker + 1), (task,)))
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5", tuple(groups), ())
    memory = SimpleNamespace(
        path_name="mte2_read",
        nodes=(
            SimpleNamespace(resource_role="l2_cache", bytes=1_000.0),
            SimpleNamespace(resource_role="dram", bytes=500.0),
        ),
    )

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=active_groups, launch_task_kind="mixed",
        resource_time_overrides_ns={}, memory_diagnostics=(memory,))

    assert finite_drain_calls == [(
        rows, columns, source_stride, active_groups,
        aggregate_descriptor_count,
    )] * (2 * active_groups)
    assert diagnostics.execution_time_s > 0.0


@pytest.mark.parametrize("task_count", (1, 2))
def test_fia_gqa_uses_source_l0_multibuffer_span(task_count):
    tensor = FiaComputeStageDemand(
        "compute_tensor", 0, 1_000_000.0, 1_000_000.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    mte2 = FiaMte2StageDemand(
        0, (FiaMte2Nd2NzDescriptorDemand(
            4, 64, 2, 1, source_row_stride_elements=64),))
    mte1 = FiaMte1StageDemand(
        0, 2_048, 2_048, FiaMte1LoadPairDemand("mm1", 64, 2))
    fixpipe = _fia_fixpipe_stage(0, count=2)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=mte2,
        mm1_mte1=mte1,
        mm1=tensor,
        mm1_fixpipe=fixpipe,
        vec1=vector,
        mm2_mte2=mte2,
        mm2_mte1=mte1,
        mm2=tensor,
        mm2_fixpipe=fixpipe,
        vec2=vector,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(
            0, (0, 1), tuple(
                dataclasses.replace(task, ordinal=ordinal)
                for ordinal in range(task_count))),),
        (),
    )

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    services = diagnostics.stage_service_s
    mm1_mte2 = services["aic_mte2_nd2nz"] / 2
    mm1_mte1 = services["aic_mte1_l1_to_l0"] / 2
    mm1_compute = services["compute_tensor"] / 2
    mm1_fixpipe = services["aic_fixpipe_l0c_to_gm"] / 2
    per_item = tuple(
        duration / 2
        for duration in (mm1_mte1, mm1_compute, mm1_fixpipe))
    expected_mm1 = mm1_mte2 + sum(per_item) + max(per_item)
    if task_count == 1:
        task_diagnostic = diagnostics.tasks[0]
        assert (
            task_diagnostic.stage_end_s["mm1_fixpipe"]
            - task_diagnostic.stage_start_s["mm1_fixpipe"]
        ) == pytest.approx(expected_mm1)
    else:
        serial_aic_span = math.fsum((
            services["aic_mte2_nd2nz"],
            services["aic_mte1_l1_to_l0"],
            services["compute_tensor"],
            services["aic_fixpipe_l0c_to_gm"],
        ))
        assert diagnostics.execution_time_s < serial_aic_span


@pytest.mark.parametrize(
    ("active_groups", "key_descriptor_count", "query_ns", "key_value_ns"),
    ((16, 2, 1_240.0, 9_520.0),
     (16, 4, 1_240.0, 14_900.0),
     (16, 8, 1_240.0, 28_220.0),
     (16, 16, 1_240.0, 56_560.0),
     (22, 2, 500.0, 10_090.0)),
)
def test_fia_mha_uses_exact_finite_mte2_chain_for_supported_program(
        active_groups, key_descriptor_count, query_ns, key_value_ns):
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    query = FiaMte2Nd2NzDescriptorDemand(
        1, 128, 2, 1, source_row_stride_elements=128)
    key_or_value = FiaMte2Nd2NzDescriptorDemand(
        256, 128, 2, key_descriptor_count,
        source_row_stride_elements=4096)
    probability = FiaMte2Nd2NzDescriptorDemand(
        1, 256, 2, 2, source_row_stride_elements=512)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=FiaMte2StageDemand(0, (query, key_or_value)),
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector,
        mm2_mte2=FiaMte2StageDemand(0, (probability, key_or_value)),
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector,
        repeat_count=2,
    )
    group = FiaCrossDomainAicGroupDemand(
        0, (0, 1), (task,),
        equivalent_aic_workers=tuple(range(1, active_groups)))
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5", (group,), ())

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=active_groups, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    probability_ns = NPU_HW.aic_mte2_nd2nz_serial_time_ns(
        512, active_groups)
    expected_per_aic_ns = (
        query_ns + key_value_ns + 4 * probability_ns + key_value_ns)

    assert diagnostics.stage_service_s["aic_mte2_nd2nz"] == pytest.approx(
        active_groups * expected_per_aic_ns * 1e-9)


def test_fia_gqa_source_order_uses_topology_not_active_aic_count():
    """A regular GQA inner program remains identical after outer repartition."""

    active_groups = 22
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    query = FiaMte2Nd2NzDescriptorDemand(
        4, 128, 2, 1, source_row_stride_elements=128)
    key_or_value = FiaMte2Nd2NzDescriptorDemand(
        256, 128, 2, 4, source_row_stride_elements=1024)
    probability = FiaMte2Nd2NzDescriptorDemand(
        4, 256, 2, 4, source_row_stride_elements=1024)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=FiaMte2StageDemand(0, (query, key_or_value)),
        mm1_mte1=FiaMte1StageDemand(
            0, 0, 0, FiaMte1LoadPairDemand("mm1", 128, 8)),
        mm1=tensor,
        mm1_fixpipe=FiaFixpipeStageDemand(
            0, (FiaFixpipeDescriptorDemand(4, 128, 128, 8),)),
        vec1=vector,
        mm2_mte2=FiaMte2StageDemand(0, (probability, key_or_value)),
        mm2_mte1=FiaMte1StageDemand(
            0, 0, 0, FiaMte1LoadPairDemand("mm2", 128, 8)),
        mm2=tensor,
        mm2_fixpipe=FiaFixpipeStageDemand(
            0, (FiaFixpipeDescriptorDemand(4, 128, 128, 1),)),
        vec2=vector,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(
            0, (0, 1), (task,),
            equivalent_aic_workers=tuple(range(1, active_groups))),),
        (),
    )

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=active_groups, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    serial_aic_span = math.fsum(
        diagnostics.stage_service_s[name] / active_groups
        for name in (
            "aic_mte2_nd2nz", "aic_mte1_l1_to_l0",
            "compute_tensor", "aic_fixpipe_l0c_to_gm",
        ))

    assert diagnostics.worker_makespan_s["aic:0"] < serial_aic_span


@pytest.mark.parametrize("has_terminal_fixpipe", (True, False))
def test_fia_gqa_outer_source_order_matches_expanded_task_program(
        has_terminal_fixpipe):
    """Persistent GQA rings must not depend on repeat compression.

    ``FiaBlockCubeNonQuantGqa`` retains its Q/KP/V and L0 ring ordinals
    across source tasks.  A compact run and the same literal tasks expanded
    one by one therefore have identical completion and service accounting.
    """

    active_groups = 22
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    query = FiaMte2Nd2NzDescriptorDemand(
        4, 128, 2, 1, source_row_stride_elements=128)
    key_or_value = FiaMte2Nd2NzDescriptorDemand(
        256, 128, 2, 4, source_row_stride_elements=1024)
    probability = FiaMte2Nd2NzDescriptorDemand(
        4, 256, 2, 4, source_row_stride_elements=1024)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=FiaMte2StageDemand(0, (query, key_or_value)),
        mm1_mte1=FiaMte1StageDemand(
            0, 0, 0, FiaMte1LoadPairDemand("mm1", 128, 8)),
        mm1=tensor,
        mm1_fixpipe=FiaFixpipeStageDemand(
            0, (FiaFixpipeDescriptorDemand(4, 128, 128, 8),)),
        vec1=vector,
        mm2_mte2=FiaMte2StageDemand(0, (probability, key_or_value)),
        mm2_mte1=FiaMte1StageDemand(
            0, 0, 0, FiaMte1LoadPairDemand("mm2", 128, 8)),
        mm2=tensor,
        mm2_fixpipe=FiaFixpipeStageDemand(
            0,
            ((FiaFixpipeDescriptorDemand(4, 128, 128, 1),)
             if has_terminal_fixpipe else ())),
        vec2=vector,
    )
    group = FiaCrossDomainAicGroupDemand(
        0, (0, 1), (task,),
        equivalent_aic_workers=tuple(range(1, active_groups)))
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5", (group,), ())
    repeated_plan = dataclasses.replace(
        plan,
        groups=(dataclasses.replace(
            group,
            tasks=(dataclasses.replace(task, repeat_count=3),),
        ),),
    )
    expanded_plan = dataclasses.replace(
        plan,
        groups=(dataclasses.replace(
            group,
            tasks=tuple(
                dataclasses.replace(task, ordinal=ordinal)
                for ordinal in range(3)
            ),
        ),),
    )

    repeated = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        repeated_plan, active_units=active_groups, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    expanded = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        expanded_plan, active_units=active_groups, launch_task_kind="mixed",
        resource_time_overrides_ns={})

    assert repeated.execution_time_s == pytest.approx(
        expanded.execution_time_s)
    assert repeated.worker_makespan_s == pytest.approx(
        expanded.worker_makespan_s)
    assert repeated.stage_service_s == pytest.approx(expanded.stage_service_s)


def test_fia_gqa_source_order_retains_short_panel_ledger():
    """A short K tail carries literal work rather than an averaged panel."""

    active_groups = 16
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    query = FiaMte2Nd2NzDescriptorDemand(
        4, 128, 2, 1, source_row_stride_elements=128)
    key_full = FiaMte2Nd2NzDescriptorDemand(
        128, 128, 2, 1, source_row_stride_elements=128,
        block_table_lookup_count=1)
    key_tail = FiaMte2Nd2NzDescriptorDemand(
        76, 128, 2, 1, source_row_stride_elements=None,
        block_table_lookup_count=1)
    probability = FiaMte2Nd2NzDescriptorDemand(
        4, 204, 2, 1, source_row_stride_elements=224)

    def panel(columns, *, full):
        mm1_l0a = 16 * 128 * 2
        mm1_l0b = math.ceil(columns / 16) * 16 * 128 * 2
        tensor_logical = float(2 * 4 * columns * 128)
        tensor_physical = float(2 * 16 * columns * 128)
        return FiaGqaSourcePanelDemand(
            source_columns=columns,
            mm1_mte1=FiaMte1StageDemand(0, mm1_l0a, mm1_l0b),
            mm1=FiaComputeStageDemand(
                "compute_tensor", 0, tensor_logical, tensor_physical),
            mm1_fixpipe=FiaFixpipeStageDemand(
                0, (() if not full else (
                    FiaFixpipeDescriptorDemand(4, 128, 224, 1),))),
            mm2_mte1=FiaMte1StageDemand(0, mm1_l0a, mm1_l0b),
            mm2=FiaComputeStageDemand(
                "compute_tensor", 0, tensor_logical, tensor_physical),
        )

    panels = (panel(128, full=True), panel(76, full=False))
    source_order = FiaGqaSourceOrderDemand(
        query_descriptor=query,
        rounds=(FiaGqaSourceRoundDemand(
            key_descriptors=(key_full, key_tail),
            probability_descriptor=probability,
            value_descriptors=(key_full, key_tail),
            panels=panels,
        ),),
    )
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=FiaMte2StageDemand(0, (query, key_full, key_tail)),
        mm1_mte1=FiaMte1StageDemand(
            0,
            sum(panel.mm1_mte1.l0a_bytes for panel in panels),
            sum(panel.mm1_mte1.l0b_bytes for panel in panels)),
        mm1=FiaComputeStageDemand(
            "compute_tensor", 0,
            sum(panel.mm1.logical_operations for panel in panels),
            sum(panel.mm1.physical_operations for panel in panels)),
        mm1_fixpipe=FiaFixpipeStageDemand(
            0, (FiaFixpipeDescriptorDemand(4, 128, 224, 1),)),
        vec1=vector,
        mm2_mte2=FiaMte2StageDemand(
            0, (probability, key_full, key_tail)),
        mm2_mte1=FiaMte1StageDemand(
            0,
            sum(panel.mm2_mte1.l0a_bytes for panel in panels),
            sum(panel.mm2_mte1.l0b_bytes for panel in panels)),
        mm2=FiaComputeStageDemand(
            "compute_tensor", 0,
            sum(panel.mm2.logical_operations for panel in panels),
            sum(panel.mm2.physical_operations for panel in panels)),
        mm2_fixpipe=FiaFixpipeStageDemand(
            0, (FiaFixpipeDescriptorDemand(4, 128, 128, 1),)),
        vec2=vector,
        gqa_source_order=source_order,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(
            0, (0, 1), (task,),
            equivalent_aic_workers=tuple(range(1, active_groups))),),
        (),
    )

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=active_groups, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    serial_aic_span = math.fsum(
        diagnostics.stage_service_s[name] / active_groups
        for name in (
            "aic_mte2_nd2nz", "aic_mte1_l1_to_l0",
            "compute_tensor", "aic_fixpipe_l0c_to_gm",
        ))

    assert diagnostics.worker_makespan_s["aic:0"] < serial_aic_span


def test_fia_gqa_source_order_separates_drain_demand_from_source_edge(
        monkeypatch):
    """Finite-drain demand must not become a literal source dependency."""

    active_groups = 16
    descriptor_count = 32
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    query = FiaMte2Nd2NzDescriptorDemand(
        4, 128, 2, 1, source_row_stride_elements=128)
    key_or_value = FiaMte2Nd2NzDescriptorDemand(
        256, 128, 2, descriptor_count,
        source_row_stride_elements=1024)
    probability = FiaMte2Nd2NzDescriptorDemand(
        4, 256, 2, descriptor_count,
        source_row_stride_elements=8192)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=FiaMte2StageDemand(0, (query, key_or_value)),
        mm1_mte1=FiaMte1StageDemand(
            0, 0, 0,
            FiaMte1LoadPairDemand("mm1", 128, 2 * descriptor_count)),
        mm1=tensor,
        mm1_fixpipe=FiaFixpipeStageDemand(
            0, (FiaFixpipeDescriptorDemand(
                4, 128, 128, 2 * descriptor_count),)),
        vec1=vector,
        mm2_mte2=FiaMte2StageDemand(0, (probability, key_or_value)),
        mm2_mte1=FiaMte1StageDemand(
            0, 0, 0,
            FiaMte1LoadPairDemand("mm2", 128, 2 * descriptor_count)),
        mm2=tensor,
        mm2_fixpipe=FiaFixpipeStageDemand(
            0, (FiaFixpipeDescriptorDemand(4, 128, 128, 1),)),
        vec2=vector,
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(
            0, (0, 1), (task,),
            equivalent_aic_workers=tuple(range(1, active_groups))),),
        (),
    )
    memory = SimpleNamespace(
        path_name="mte2_read",
        nodes=(
            SimpleNamespace(resource_role="l2_cache", bytes=1_000.0),
            SimpleNamespace(resource_role="dram", bytes=500.0),
        ),
    )
    baseline = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=active_groups, launch_task_kind="mixed",
        resource_time_overrides_ns={}, memory_diagnostics=(memory,))

    finite_calls = []
    original_finite_time = NPU_HW.aic_mte2_fia_gqa_finite_drain_time_ns

    def doubled_finite_time(
            rows, columns, source_row_stride_elements,
            observed_active_groups, observed_descriptor_count):
        finite_calls.append((
            rows, columns, source_row_stride_elements,
            observed_active_groups, observed_descriptor_count))
        return 2.0 * original_finite_time(
            rows, columns, source_row_stride_elements,
            observed_active_groups, observed_descriptor_count)

    monkeypatch.setattr(
        NPU_HW, "aic_mte2_fia_gqa_finite_drain_time_ns",
        doubled_finite_time)
    doubled = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=active_groups, launch_task_kind="mixed",
        resource_time_overrides_ns={}, memory_diagnostics=(memory,))

    assert (256, 128, 1024, active_groups, descriptor_count) in finite_calls
    assert doubled.stage_service_s["aic_mte2_nd2nz"] > (
        baseline.stage_service_s["aic_mte2_nd2nz"])
    assert doubled.execution_time_s == pytest.approx(
        baseline.execution_time_s)


def test_split_fuse_mte2_uses_serial_fill_and_stream_service(monkeypatch):
    serial_calls = []
    stream_calls = []
    original_serial_time = NPU_HW.aic_mte2_nd2nz_serial_time_ns
    original_stream_time = NPU_HW.aic_mte2_nd2nz_stream_time_ns
    original_source_order = formula_module._split_fuse_mm2_source_order_time

    def record_serial_time(descriptor_bytes, active_aic_groups):
        serial_calls.append((descriptor_bytes, active_aic_groups))
        return original_serial_time(descriptor_bytes, active_aic_groups)

    def record_stream_time(descriptor_bytes, active_aic_groups):
        stream_calls.append((descriptor_bytes, active_aic_groups))
        return original_stream_time(descriptor_bytes, active_aic_groups)

    source_order_calls = []

    def record_source_order(*args, **kwargs):
        source_order_calls.append((args, kwargs))
        return original_source_order(*args, **kwargs)

    monkeypatch.setattr(
        NPU_HW, "aic_mte2_nd2nz_serial_time_ns", record_serial_time)
    monkeypatch.setattr(
        NPU_HW, "aic_mte2_nd2nz_stream_time_ns", record_stream_time)
    monkeypatch.setattr(
        formula_module, "_split_fuse_mm2_source_order_time",
        record_source_order)
    plan = FiaSplitFuseWorkPlan(
        mechanism_id="cann_split_fuse_aic_dual_aiv_v1",
        aic_worker_count=24,
        tile_q=128,
        kv_stack_ceiling=512,
        head_dim=128,
        query_head_count=24,
        kv_head_count=24,
        request_runs=(FiaSplitFuseRequestRunDemand(128, 512, 1),),
        aic_l2_bytes=1.0,
        aiv_l2_bytes=1.0,
        aiv_mte3_bytes=1.0,
        paged_kv_cache=False,
        kv_page_size=None,
    )

    diagnostics = NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        plan,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )

    assert diagnostics.execution_time_s > 0.0
    assert len(source_order_calls) == 1
    assert serial_calls.count((65536, 24)) == 1
    assert stream_calls.count((65536, 24)) == 1
    assert (32768, 24) in stream_calls
    assert (65536, 24) in stream_calls

    source_order_call_count = len(source_order_calls)
    paged_plan = dataclasses.replace(
        plan, paged_kv_cache=True, kv_page_size=128)
    paged_diagnostics = NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        paged_plan,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )
    assert len(source_order_calls) == source_order_call_count
    lookup_s = NPU_HW.resource_by_role(
        "compute_fma").exact_request_service_time_s(
            "aic_scalar_global_get_value_serialized", 1)
    assert paged_plan.paged_descriptor_lookup_count(512) == 8
    assert paged_diagnostics.stage_service_s[
        "split_fuse_paged_kv_lookup"] == pytest.approx(
            paged_plan.task_count * 8 * lookup_s)
    assert paged_diagnostics.execution_time_s > diagnostics.execution_time_s

    # ``PRE_LAUNCH=2`` in the pinned SplitFuse source permits a V
    # prefix only when two later stacks remain in the outer QK loop.  A
    # one-stack drain retains the old aggregate completion, while a four-stack
    # task may overlap the V prefix with the existing Vec1 wait without
    # changing any PV service total.
    one_stack_prefix = dataclasses.replace(
        paged_plan, pv_v_ingress_before_softmax=True)
    one_stack_diagnostics = NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        one_stack_prefix,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )
    assert one_stack_diagnostics.execution_time_s == pytest.approx(
        paged_diagnostics.execution_time_s)
    four_stack_plan = dataclasses.replace(
        paged_plan,
        request_runs=(FiaSplitFuseRequestRunDemand(128, 2048, 1),),
    )
    four_stack_diagnostics = NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        four_stack_plan,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )
    four_stack_prefix = dataclasses.replace(
        four_stack_plan, pv_v_ingress_before_softmax=True)
    four_stack_prefix_diagnostics = (
        NPU_FORMULA._resolve_fia_split_fuse_work_plan(
            four_stack_prefix,
            active_units=24,
            launch_task_kind="mixed",
            resource_time_overrides_ns={},
        ))
    assert four_stack_prefix_diagnostics.execution_time_s < (
        four_stack_diagnostics.execution_time_s)
    assert four_stack_prefix_diagnostics.stage_service_s[
        "split_fuse_mm2"] == pytest.approx(
            four_stack_diagnostics.stage_service_s["split_fuse_mm2"])

    # BlockMmadPV loads V before softmaxReady in both the paged and
    # contiguous template specializations.  The former source restriction to
    # paged launches would therefore serialize a legal contiguous prefix.
    contiguous_four_stack = dataclasses.replace(
        plan,
        request_runs=(FiaSplitFuseRequestRunDemand(128, 2048, 1),),
        pv_v_ingress_before_softmax=True)
    contiguous_without_prefix = dataclasses.replace(
        contiguous_four_stack, pv_v_ingress_before_softmax=False)
    assert NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        contiguous_four_stack,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    ).execution_time_s < NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        contiguous_without_prefix,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    ).execution_time_s


def test_split_fuse_shared_services_exclude_taskless_launch_groups(
        monkeypatch):
    observed_active_groups = []
    original_serial_time = NPU_HW.aic_mte2_nd2nz_serial_time_ns
    original_stream_time = NPU_HW.aic_mte2_nd2nz_stream_time_ns

    def record_serial_time(descriptor_bytes, active_aic_groups):
        observed_active_groups.append(active_aic_groups)
        return original_serial_time(descriptor_bytes, active_aic_groups)

    def record_stream_time(descriptor_bytes, active_aic_groups):
        observed_active_groups.append(active_aic_groups)
        return original_stream_time(descriptor_bytes, active_aic_groups)

    monkeypatch.setattr(
        NPU_HW, "aic_mte2_nd2nz_serial_time_ns", record_serial_time)
    monkeypatch.setattr(
        NPU_HW, "aic_mte2_nd2nz_stream_time_ns", record_stream_time)
    plan = FiaSplitFuseWorkPlan(
        mechanism_id="cann_split_fuse_aic_dual_aiv_v1",
        aic_worker_count=24,
        tile_q=128,
        kv_stack_ceiling=512,
        head_dim=128,
        query_head_count=32,
        kv_head_count=8,
        request_runs=(FiaSplitFuseRequestRunDemand(32, 4096, 1),),
        aic_l2_bytes=1.0,
        aiv_l2_bytes=1.0,
        aiv_mte3_bytes=1.0,
        paged_kv_cache=True,
        kv_page_size=128,
        pv_v_ingress_before_softmax=True,
    )

    diagnostics = NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        plan,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )

    assert plan.task_counts == (1,) * 8 + (0,) * 16
    assert observed_active_groups
    assert set(observed_active_groups) == {8}
    assert len(tuple(
        task for task in diagnostics.tasks
        if task.task_name.endswith("compressed_program"))) == 8


def test_split_fuse_local_l2_service_ignores_aggregate_domain_mix():
    plan = FiaSplitFuseWorkPlan(
        mechanism_id="cann_split_fuse_aic_dual_aiv_v1",
        aic_worker_count=24,
        tile_q=128,
        kv_stack_ceiling=512,
        head_dim=128,
        query_head_count=32,
        kv_head_count=8,
        request_runs=(FiaSplitFuseRequestRunDemand(128, 4096, 1),),
        aic_l2_bytes=9.0,
        aiv_l2_bytes=1.0,
        aiv_mte3_bytes=1.0,
        paged_kv_cache=True,
        kv_page_size=128,
        pv_v_ingress_before_softmax=True,
    )

    aic_heavy = NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        plan,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )
    aiv_heavy = NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        dataclasses.replace(plan, aic_l2_bytes=1.0, aiv_l2_bytes=9.0),
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )

    # Aggregate AIC/AIV bytes constrain the outer shared-L2 bound.  They do
    # not partition physical bandwidth inside a local source stage, where
    # only that stage's active issuers contend for the shared capacity.
    assert aiv_heavy.stage_service_s == pytest.approx(
        aic_heavy.stage_service_s)
    assert aiv_heavy.execution_time_s == pytest.approx(
        aic_heavy.execution_time_s)


def test_split_fuse_mm2_source_order_conserves_demand_and_l1_slots():
    descriptor_durations = iter((
        5.0, 18.0, 1.0, 7.0, 9.0, 6.0,
        7.0, 15.0, 19.0, 18.0, 4.0, 19.0,
    ))
    tensor_durations = iter((
        9.0, 19.0, 14.0, 16.0, 2.0, 18.0, 15.0, 4.0,
    ))
    descriptors = []
    panels = []

    def descriptor_duration(byte_count, ordinal):
        descriptors.append((byte_count, ordinal))
        return next(descriptor_durations)

    def tensor_duration(panel_columns):
        panels.append(panel_columns)
        return next(tensor_durations)

    completion = _split_fuse_mm2_source_order_time(
        (512, 512), 128, 128,
        descriptor_duration, tensor_duration, 0.0)

    expected_chunk = ((65536, 0), (32768, 1), (32768, 2))
    assert tuple(byte_count for byte_count, _ in descriptors) == (
        tuple(byte_count for byte_count, _ in expected_chunk) * 4)
    assert tuple(ordinal for _, ordinal in descriptors) == tuple(range(12))
    assert sum(byte_count for byte_count, _ in descriptors) == (
        2 * 128 * 1024 + 2 * 1024 * 128)
    assert panels == [128] * 8
    # The third and fourth chunks must wait for their alternating L1 slots.
    # Removing that source constraint gives 132 for this synthetic program.
    assert completion == pytest.approx(137.0)


def test_split_fuse_does_not_charge_unimplemented_runtime_queue_lifetimes():
    plan = FiaSplitFuseWorkPlan(
        mechanism_id="cann_split_fuse_aic_dual_aiv_v1",
        aic_worker_count=24,
        tile_q=128,
        kv_stack_ceiling=512,
        head_dim=128,
        query_head_count=24,
        kv_head_count=24,
        request_runs=(FiaSplitFuseRequestRunDemand(128, 1024, 1),),
        aic_l2_bytes=1.0,
        aiv_l2_bytes=1.0,
        aiv_mte3_bytes=1.0,
        paged_kv_cache=False,
        kv_page_size=None,
    )

    diagnostics = NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        plan,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )

    # The pinned SplitFuse regular, online-softmax, and rescale source
    # programs use LocalTensor buffers and do not execute a runtime TQue
    # lifecycle. Generic AIV queue-control evidence cannot be charged here.
    assert diagnostics.stage_service_s[
        "split_fuse_vec1_queue_control"] == 0.0
    assert diagnostics.stage_service_s[
        "split_fuse_vec2_queue_control"] == 0.0
    assert diagnostics.stage_service_s[
        "fia_output_init_syncall"] == pytest.approx(
            NPU_HW.fia_aiv_sync_all_time_ns(48) * 1e-9)


def test_split_fuse_vec1_prices_score_mask_and_row_state_by_source_scope():
    plan = FiaSplitFuseWorkPlan(
        mechanism_id="cann_split_fuse_aic_dual_aiv_v1",
        aic_worker_count=24,
        tile_q=128,
        kv_stack_ceiling=512,
        head_dim=128,
        query_head_count=24,
        kv_head_count=24,
        request_runs=(FiaSplitFuseRequestRunDemand(128, 1024, 1),),
        aic_l2_bytes=1.0,
        aiv_l2_bytes=1.0,
        aiv_mte3_bytes=1.0,
        paged_kv_cache=False,
        kv_page_size=None,
    )

    diagnostics = NPU_FORMULA._resolve_fia_split_fuse_work_plan(
        plan,
        active_units=24,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )

    # For this two-stack source program, stack zero has only ScaleS,
    # CalcExp::Sub/Exp and DownCastP score work.  Stack one additionally has
    # row-wise UpdateGlobalRowMax/RowSum state work and causal ApplyMask.
    # This guards against restoring the old unconditional second score-wise
    # generic operation in every stack.
    assert diagnostics.stage_service_s["split_fuse_vec1"] == pytest.approx(
        0.00037796056831734636)
    # Includes the measured three-handoff per-depth cross-core edges
    # (24 groups: 1841.9 ns for this single task) and the sequential task
    # boundary gate of the pinned source task loop.
    assert diagnostics.execution_time_s == pytest.approx(
        3.2261227163756164e-05)


def test_fia_mla_multibuffer_only_changes_homogeneous_steady_state():
    tensor = FiaComputeStageDemand(
        "compute_tensor", 0, 1_000_000.0, 1_000_000.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    peer_vector = FiaComputeStageDemand("compute_vector", 1, 0.0, 0.0)
    serial_task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=_fia_mte2_stage(0),
        mm1_mte1=FiaMte1StageDemand(0, 65_536, 65_536),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0, count=1),
        vec1=vector,
        mm2_mte2=_fia_mte2_stage(0),
        mm2_mte1=FiaMte1StageDemand(0, 65_536, 65_536),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0, count=1),
        vec2=vector,
        vec1_peer=peer_vector,
        vec2_peer=peer_vector,
    )

    def resolve(task):
        plan = FiaCrossDomainWorkPlan(
            "cann_fia_1_2_three_handoff_v5",
            (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
            (),
        )
        return NPU_FORMULA._resolve_fia_cross_domain_work_plan(
            plan, active_units=1, launch_task_kind="mixed",
            resource_time_overrides_ns={})

    single_serial = resolve(serial_task)
    single_multibuffer = resolve(dataclasses.replace(
        serial_task,
        aic_pipeline_kind="fia_block_cube_nonquant_mla_multibuffer",
    ))
    queue_control_cycle_s = NPU_HW.resource_by_role(
        "scalar_aiv").exact_request_service_time_s(
            "aiv_queue_control_cycle", 1)
    assert single_serial.stage_service_s["aiv_queue_control"] == (
        pytest.approx(8 * queue_control_cycle_s))
    assert single_multibuffer.stage_service_s["aiv_queue_control"] == 0.0

    repeated_serial = resolve(dataclasses.replace(
        serial_task, repeat_count=8))
    repeated_multibuffer = resolve(dataclasses.replace(
        serial_task,
        aic_pipeline_kind="fia_block_cube_nonquant_mla_multibuffer",
        repeat_count=8,
    ))
    expanded_multibuffer = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        FiaCrossDomainWorkPlan(
            "cann_fia_1_2_three_handoff_v5",
            (FiaCrossDomainAicGroupDemand(
                0,
                (0, 1),
                tuple(dataclasses.replace(
                    serial_task,
                    ordinal=ordinal,
                    aic_pipeline_kind=(
                        "fia_block_cube_nonquant_mla_multibuffer"),
                ) for ordinal in range(8)),
            ),),
            (),
        ),
        active_units=1,
        launch_task_kind="mixed",
        resource_time_overrides_ns={},
    )
    assert repeated_multibuffer.execution_time_s < (
        repeated_serial.execution_time_s)
    assert repeated_multibuffer.execution_time_s == pytest.approx(
        expanded_multibuffer.execution_time_s)
    assert repeated_serial.stage_service_s["aiv_queue_control"] == (
        pytest.approx(8 * 8 * queue_control_cycle_s))
    assert repeated_multibuffer.stage_service_s["aiv_queue_control"] == 0.0
    assert repeated_multibuffer.stage_service_s[
        "fia_mla_multitask_dependency_startup"] == pytest.approx(870.0e-9)
    assert {
        key: value
        for key, value in repeated_multibuffer.stage_service_s.items()
        if key not in {
            "aiv_queue_control", "fia_mla_multitask_dependency_startup"}
    } == {
        key: value
        for key, value in repeated_serial.stage_service_s.items()
        if key not in {
            "aiv_queue_control", "fia_mla_multitask_dependency_startup"}
    }


def test_fia_mla_source_order_keeps_the_second_k_half_literal():
    """CANN MLA emits K/RoPE before K/no-PE in its second 288-K half."""

    from evaluation.kernel.observation_normalizer import normalize_kernel_spec
    from evaluation.kernel.profiler_observation import extract_ncu_metrics
    from src.kernel.predictors.flash_attn import FlashAttentionPredictor

    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "kernel_dataset/ascend_910b/flash_attn.json.zst")
    dataset = load_kernel_dataset_document(dataset_path)
    entry = dataset["FLASHATTN_MLA_B64_S2048_HQ64_HKV1_D512_C0"]
    observation = entry["kernels"][0]
    kernel = normalize_kernel_spec(
        entry["op_info"], observation, "ascend_910b", NPU_HW,
        kernel_ordinal=0,
        profiler_metrics=extract_ncu_metrics(observation),
    )
    prediction_input, _ = FlashAttentionPredictor().resource_input(
        kernel, NPU_HW)
    plan = prediction_input.pipeline_plan
    assert plan is not None
    source_task = next(
        task
        for group in plan.groups
        for task in group.tasks
        if (task.mla_source_order is not None
            and len(task.mla_source_order.mm1_descriptors) == 19))
    source_pairs = tuple(
        (descriptor.columns, descriptor.source_row_stride_elements)
        for descriptor in source_task.mla_source_order.mm1_descriptors[3:])
    assert source_pairs == ((256, 512), (32, 64), (32, 64), (256, 512)) * 4

    prediction = NPU_FORMULA.predict_kernel(prediction_input)
    assert prediction.t_total_s > 0.0
    assert prediction.diagnostics.pipeline is not None
    assert prediction.diagnostics.pipeline.stage_service_s[
        "aic_mte2_nd2nz"] > 0.0


def test_oracle_replacement_rescales_regular_gqa_source_panels():
    """Oracle Cube replacement must retain the literal GQA panel ledger."""

    from evaluation.kernel.observation_normalizer import normalize_kernel_spec
    from evaluation.kernel.profiler_observation import extract_ncu_metrics
    from src.kernel.predictors.flash_attn import FlashAttentionPredictor

    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "kernel_dataset/ascend_910b/flash_attn.json.zst")
    dataset = load_kernel_dataset_document(dataset_path)
    entry = dataset["FLASHATTN_B16_S1024_HQ32_HKV32_D128_C0"]
    observation = entry["kernels"][0]
    kernel = normalize_kernel_spec(
        entry["op_info"], observation, "ascend_910b", NPU_HW,
        kernel_ordinal=0,
        profiler_metrics=extract_ncu_metrics(observation),
    )
    prediction_input, _ = FlashAttentionPredictor().resource_input(
        kernel, NPU_HW)
    plan = prediction_input.pipeline_plan
    assert isinstance(plan, FiaCrossDomainWorkPlan)
    original_task = next(
        task
        for group in plan.groups
        for task in group.tasks
        if task.gqa_source_order is not None)
    original_panel = original_task.gqa_source_order.rounds[0].panels[0]
    replacement = (
        prediction_input.demand_per_resource["compute_tensor"] / 2.0)

    replaced = replace_structured_demands(
        prediction_input, {"compute_tensor": replacement})

    replaced_plan = replaced.pipeline_plan
    assert isinstance(replaced_plan, FiaCrossDomainWorkPlan)
    replaced_task = next(
        task
        for group in replaced_plan.groups
        for task in group.tasks
        if task.gqa_source_order is not None)
    replaced_panel = replaced_task.gqa_source_order.rounds[0].panels[0]
    assert replaced_task.mm1.logical_operations == pytest.approx(
        original_task.mm1.logical_operations / 2.0)
    assert replaced_task.mm2.physical_operations == pytest.approx(
        original_task.mm2.physical_operations / 2.0)
    assert replaced_panel.mm1.logical_operations == pytest.approx(
        original_panel.mm1.logical_operations / 2.0)
    assert replaced_panel.mm2.physical_operations == pytest.approx(
        original_panel.mm2.physical_operations / 2.0)
    for aggregate, panels in (
            (replaced_task.mm1,
             tuple(panel.mm1 for round_demand in replaced_task.gqa_source_order.rounds
                   for panel in round_demand.panels)),
            (replaced_task.mm2,
             tuple(panel.mm2 for round_demand in replaced_task.gqa_source_order.rounds
                   for panel in round_demand.panels))):
        assert math.fsum(panel.logical_operations for panel in panels) == (
            pytest.approx(aggregate.logical_operations))
        assert math.fsum(panel.physical_operations for panel in panels) == (
            pytest.approx(aggregate.physical_operations))


def test_fia_vec1_prices_source_row_sum_on_the_aiv_timeline():
    tensor = FiaComputeStageDemand("compute_tensor", 0, 0.0, 0.0)
    vector = FiaComputeStageDemand("compute_vector", 0, 0.0, 0.0)
    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=_fia_mte2_stage(0),
        mm1_mte1=_fia_mte1_stage(0),
        mm1=tensor,
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector,
        mm2_mte2=_fia_mte2_stage(0),
        mm2_mte1=_fia_mte1_stage(0),
        mm2=tensor,
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector,
        vec1_block_reduce_sum=(FiaBlockReduceSumDescriptorDemand(
            row_chunk_round=8,
            column_count=512,
            column_alignment=32,
        ),),
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (),
    )
    without_row_sum = dataclasses.replace(
        plan,
        groups=(dataclasses.replace(
            plan.groups[0],
            tasks=(dataclasses.replace(
                task, vec1_block_reduce_sum=()),),
        ),),
    )

    diagnostics = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        plan, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    baseline = NPU_FORMULA._resolve_fia_cross_domain_work_plan(
        without_row_sum, active_units=1, launch_task_kind="mixed",
        resource_time_overrides_ns={})
    row_sum_s = NPU_HW.fia_block_reduce_sum_time_ns(8, 512, 32) * 1e-9
    row_max_s = NPU_HW.fia_block_reduce_max_time_ns(8, 512, 32) * 1e-9
    brcb_s = NPU_HW.fia_brcb_time_ns(8) * 1e-9
    source_row_reduction_s = row_sum_s + row_max_s + brcb_s
    queue_control_cycle_s = NPU_HW.resource_by_role(
        "scalar_aiv").exact_request_service_time_s(
            "aiv_queue_control_cycle", 1)

    assert row_sum_s == pytest.approx(96.2646484375e-9)
    assert diagnostics.stage_service_s["aiv_block_reduce_sum"] == (
        pytest.approx(row_sum_s))
    assert diagnostics.stage_service_s["aiv_block_reduce_max"] == (
        pytest.approx(row_max_s))
    assert diagnostics.stage_service_s["aiv_brcb"] == pytest.approx(brcb_s)
    assert diagnostics.stage_service_s["aiv_queue_control"] == (
        pytest.approx(4 * queue_control_cycle_s))
    assert diagnostics.stage_service_s["compute_vector"] == pytest.approx(
        source_row_reduction_s)
    assert (
        diagnostics.tasks[0].stage_end_s["vec1"]
        - baseline.tasks[0].stage_end_s["vec1"]
    ) == pytest.approx(source_row_reduction_s)


def test_fia_oracle_overrides_constrain_whole_execution_domains():
    def vector_stage(worker, operations):
        return FiaComputeStageDemand(
            "compute_vector", worker, operations, operations,
            (ComputeServiceDemand("generic", operations),))

    task = FiaCrossDomainTaskDemand(
        ordinal=0,
        mm1_mte2=_fia_mte2_stage(0),
        mm1_mte1=_fia_mte1_stage(0),
        mm1=FiaComputeStageDemand("compute_tensor", 0, 10.0, 10.0),
        mm1_fixpipe=_fia_fixpipe_stage(0),
        vec1=vector_stage(0, 5.0),
        mm2_mte2=_fia_mte2_stage(0),
        mm2_mte1=_fia_mte1_stage(0),
        mm2=FiaComputeStageDemand("compute_tensor", 0, 10.0, 10.0),
        mm2_fixpipe=_fia_fixpipe_stage(0),
        vec2=vector_stage(0, 5.0),
    )
    plan = FiaCrossDomainWorkPlan(
        "cann_fia_1_2_three_handoff_v5",
        (FiaCrossDomainAicGroupDemand(0, (0, 1), (task,)),),
        (vector_stage(0, 4.0),),
    )
    base = _make_input(
        contract_hardware=NPU_HW,
        active_units=1,
        grid_size=1,
        launch_task_kind="mixed",
        demand_per_resource={
            "compute_tensor": 20.0,
            "compute_vector": 14.0,
        },
        include_dispatch_overhead=False,
    )
    terminal_s = 4.0 / NPU_HW.resource_by_role(
        "compute_vector").service_rate_for_one_unit("generic")
    terminal_prelude_s = (
        NPU_HW.fia_flash_decode_terminal_prelude_time_ns(
            plan.active_group_count) * 1e-9)
    typed_compute_demands = []
    for demand in base.compute_demands:
        if demand.resource_role == "compute_tensor":
            typed_compute_demands.append(demand)
        elif demand.resource_role == "compute_vector":
            typed_compute_demands.append(dataclasses.replace(
                demand,
                service_demands=(ComputeServiceDemand(
                    "generic", 14.0,
                    operations_by_worker=(14.0, 0.0),
                    serial_invocations_by_worker=(0, 0),
                ),),
            ))
        else:
            raise AssertionError(demand.resource_role)
    typed_compute_demands = tuple(typed_compute_demands)

    tensor_override_ns = 1_000_000.0
    tensor_prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
        base,
        compute_demands=typed_compute_demands,
        pipeline_plan=plan,
        support_level="partial",
        resource_time_overrides_ns={
            "compute_tensor": tensor_override_ns,
        },
    ))
    assert tensor_prediction.diagnostics.pipeline.execution_time_s == (
        pytest.approx(
            tensor_override_ns * 1e-9 + terminal_prelude_s + terminal_s))
    assert tensor_prediction.diagnostics.pipeline.critical_path[0] == (
        "resource_time_override:compute_tensor")

    vector_override_ns = 2_000_000.0
    vector_prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
        base,
        compute_demands=typed_compute_demands,
        pipeline_plan=plan,
        support_level="partial",
        resource_time_overrides_ns={
            "compute_tensor": tensor_override_ns,
            "compute_vector": vector_override_ns,
        },
    ))
    assert vector_prediction.diagnostics.pipeline.execution_time_s == (
        pytest.approx(vector_override_ns * 1e-9))
    assert vector_prediction.diagnostics.pipeline.critical_path == (
        "resource_time_override:compute_vector",)


def test_gpu_source_does_not_invent_workload_request_independence():
    producers, requests_per_producer = gpu_memory_source(C500_HW)

    assert producers == 1.0
    assert requests_per_producer is None


def test_pipeline_stage_requires_exactly_one_demand_representation():
    with pytest.raises(
            ValueError, match="fractional pipeline stages require"):
        PipelineStageDemand("load", "memory:read", None)
    with pytest.raises(
            ValueError, match="must not declare a work_fraction"):
        PipelineStageDemand(
            "compute", "compute:compute_vector", 0.5,
            compute_service_demands=(
                ComputeServiceDemand("generic", 1.0),),
        )
    with pytest.raises(
            ValueError, match="require a compute service"):
        PipelineStageDemand(
            "compute", "memory:read", None,
            compute_service_demands=(
                ComputeServiceDemand("generic", 1.0),),
        )
    with pytest.raises(ValueError, match="exactly one local worker"):
        PipelineStageDemand(
            "compute", "compute:compute_vector", None,
            compute_exact_request_demands=(ComputeExactRequestDemand(
                "norm_sqrt_duplicate_div_serialized", ((1,), (1,))),),
        )
    with pytest.raises(ValueError, match="cannot mix compute and memory"):
        PipelineStageDemand(
            "mixed", "compute:compute_vector", None,
            compute_service_demands=(
                ComputeServiceDemand("generic", 1.0),),
            memory_node_demands=(
                PhaseMemoryDemand("read", "dram", 1.0),),
        )
    with pytest.raises(ValueError, match="must belong to service path"):
        PipelineStageDemand(
            "load", "memory:read", None,
            memory_node_demands=(
                PhaseMemoryDemand("other", "dram", 1.0),),
        )
    with pytest.raises(ValueError, match="cannot repeat a path node"):
        PipelineStageDemand(
            "load", "memory:read", None,
            memory_node_demands=(
                PhaseMemoryDemand("read", "dram", 1.0),),
            launch_shared_memory_node_demands=(
                PhaseMemoryDemand("read", "dram", 1.0),),
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        PipelineStageDemand(
            "load", "memory:read", None,
            memory_node_demands=(
                PhaseMemoryDemand("read", "dram", 1.0),),
            completion_resource_roles=(),
        )
    with pytest.raises(ValueError, match="must belong to the stage"):
        PipelineStageDemand(
            "load", "memory:read", None,
            memory_node_demands=(
                PhaseMemoryDemand("read", "dram", 1.0),),
            completion_resource_roles=("l2_cache",),
        )
    with pytest.raises(ValueError, match="require memory node demands"):
        PipelineStageDemand(
            "load", "memory:read", 1.0,
            completion_resource_roles=("dram",),
        )
    with pytest.raises(
            ValueError, match="requires absolute memory node demands"):
        PipelineStageDemand(
            "load", "memory:read", 1.0,
            local_memory_source_completion="sequence_start",
        )
    with pytest.raises(
            ValueError, match="cannot include first-request latency"):
        PipelineStageDemand(
            "load", "memory:read", None,
            include_first_request_latency=True,
            memory_node_demands=(
                PhaseMemoryDemand("read", "dram", 1.0),),
            local_memory_source_completion="sequence_start",
        )
    with pytest.raises(ValueError, match="exact memory requests require"):
        PipelineStageDemand(
            "compute", "compute:compute_vector", 1.0,
            memory_exact_request_demands=(MemoryExactRequestDemand(
                "mte2", "aiv_mte2_request_completion", ((1024,),)),),
        )
    with pytest.raises(ValueError, match="exactly one local worker"):
        PipelineStageDemand(
            "load", "memory:read", 1.0,
            memory_exact_request_demands=(MemoryExactRequestDemand(
                "mte2", "aiv_mte2_request_completion",
                ((1024,), (1024,))),),
        )
    with pytest.raises(ValueError, match="requires exactly one exact"):
        PipelineStageDemand(
            "load", "memory:read", None,
            memory_node_demands=(
                PhaseMemoryDemand("read", "mte2", 1024.0),),
            memory_exact_request_replaces_resource_capacity=True,
        )
    with pytest.raises(ValueError, match="requires bytes on the exact"):
        PipelineStageDemand(
            "load", "memory:read", None,
            memory_node_demands=(
                PhaseMemoryDemand("read", "dram", 1024.0),),
            memory_exact_request_demands=(MemoryExactRequestDemand(
                "mte2", "aiv_mte2_request_completion", ((1024,),)),),
            memory_exact_request_replaces_resource_capacity=True,
        )
    with pytest.raises(ValueError, match="require a mechanism service"):
        PipelineStageDemand(
            "completion", "compute:compute_tensor", None,
            absolute_duration_s=1.0e-6,
        )
    with pytest.raises(ValueError, match="cannot mix aggregate or exact"):
        PipelineStageDemand(
            "completion", "mechanism:test_completion", 1.0,
            absolute_duration_s=1.0e-6,
        )
    with pytest.raises(ValueError, match="cannot include first-request"):
        PipelineStageDemand(
            "completion", "mechanism:test_completion", None, True,
            absolute_duration_s=1.0e-6,
        )

    exact = PipelineStageDemand(
        "compute", "compute:compute_vector", None,
        compute_exact_request_demands=(ComputeExactRequestDemand(
            "norm_sqrt_duplicate_div_serialized", ((1,),)),),
    )
    memory = PipelineStageDemand(
        "load", "memory:read", None,
        memory_node_demands=(
            PhaseMemoryDemand("read", "l2_cache", 2.0),),
        launch_shared_memory_node_demands=(
            PhaseMemoryDemand("read", "dram", 1.0),
        ),
        completion_resource_roles=("l2_cache",),
        local_memory_source_completion="sequence_start",
    )
    fractional = PipelineStageDemand("load", "memory:read", 1.0)
    exact_memory = PipelineStageDemand(
        "load", "memory:read", 1.0,
        memory_exact_request_demands=(MemoryExactRequestDemand(
            "mte2", "aiv_mte2_request_completion", ((1024,),)),),
    )
    absolute = PipelineStageDemand(
        "completion", "mechanism:test_completion", None,
        absolute_duration_s=1.0e-6,
    )
    assert exact.work_fraction is None
    assert memory.work_fraction is None
    assert memory.completion_resource_roles == ("l2_cache",)
    assert memory.local_memory_source_completion == "sequence_start"
    assert fractional.memory_node_demands == ()
    assert fractional.launch_shared_memory_node_demands == ()
    assert exact_memory.memory_exact_request_demands[0].resource_role == (
        "mte2")
    assert not exact_memory.memory_exact_request_replaces_resource_capacity
    assert absolute.absolute_duration_s == pytest.approx(1.0e-6)


def test_pipeline_stage_template_binding_rechecks_only_dynamic_scalar():
    fractional_template = PipelineStageDemand(
        "load", "memory:read", 1.0, predecessors=(),
        prior_resources=("resource:prior",))
    bound_fractional = PipelineStageDemand.bind_validated_dynamic_value(
        fractional_template, work_fraction=0.25)
    assert bound_fractional.work_fraction == pytest.approx(0.25)
    assert bound_fractional.name == fractional_template.name
    assert bound_fractional.predecessors == fractional_template.predecessors
    assert bound_fractional.prior_resources == fractional_template.prior_resources
    with pytest.raises(ValueError, match="cannot exceed one"):
        PipelineStageDemand.bind_validated_dynamic_value(
            fractional_template, work_fraction=1.1)
    with pytest.raises(ValueError, match="require exactly one work fraction"):
        PipelineStageDemand.bind_validated_dynamic_value(fractional_template)

    absolute_template = PipelineStageDemand(
        "completion", "mechanism:test_completion", None,
        absolute_duration_s=1.0e-6)
    bound_absolute = PipelineStageDemand.bind_validated_dynamic_value(
        absolute_template, absolute_duration_s=2.0e-6)
    assert bound_absolute.absolute_duration_s == pytest.approx(2.0e-6)
    with pytest.raises(TypeError, match="finite number"):
        PipelineStageDemand.bind_validated_dynamic_value(
            absolute_template, absolute_duration_s=float("nan"))


def test_shared_pipeline_stage_template_binds_hardware_adapters_together(
        monkeypatch):
    import src.kernel.predictors.common as common_module

    common_module._pipeline_stage_templates.clear()
    original_bind = PipelineStageDemand.bind_validated_dynamic_value
    template_ids = []

    def record_template(template, **kwargs):
        template_ids.append(id(template))
        return original_bind(template, **kwargs)

    monkeypatch.setattr(
        PipelineStageDemand,
        "bind_validated_dynamic_value",
        record_template)
    c500_stage = bind_pipeline_stage_template(
        "load_k", "memory:global_read", 0.25, True,
        predecessors=("load_q",))
    a800_stage = bind_pipeline_stage_template(
        "load_k", "memory:global_read", 0.5, True,
        predecessors=("load_q",))

    assert c500_stage.work_fraction == pytest.approx(0.25)
    assert a800_stage.work_fraction == pytest.approx(0.5)
    assert template_ids[0] == template_ids[1]
    assert len(common_module._pipeline_stage_templates) == 1
    common_module._pipeline_stage_templates.clear()


def test_shared_typed_compute_stage_template_rebinds_dynamic_operations(
        monkeypatch):
    import src.kernel.predictors.common as common_module

    common_module._typed_compute_pipeline_stage_templates.clear()
    original_bind = PipelineStageDemand.bind_validated_typed_compute_operations
    template_ids = []

    def record_template(template, operations):
        template_ids.append(id(template))
        return original_bind(template, operations)

    monkeypatch.setattr(
        PipelineStageDemand,
        "bind_validated_typed_compute_operations",
        record_template)
    c500_stage = bind_typed_compute_pipeline_stage_template(
        "source_sfu", "compute:compute_sfu",
        ("exp_dependency", "reciprocal_one_plus_dependency"),
        (8.0, 8.0))
    a800_stage = bind_typed_compute_pipeline_stage_template(
        "source_sfu", "compute:compute_sfu",
        ("exp_dependency", "reciprocal_one_plus_dependency"),
        (4.0, 4.0))

    assert tuple(demand.operations for demand in c500_stage.compute_service_demands) == (
        8.0, 8.0)
    assert tuple(demand.operations for demand in a800_stage.compute_service_demands) == (
        4.0, 4.0)
    assert template_ids[0] == template_ids[1]
    assert len(common_module._typed_compute_pipeline_stage_templates) == 1
    common_module._typed_compute_pipeline_stage_templates.clear()


def test_shared_memory_access_template_rechecks_only_dynamic_bytes(monkeypatch):
    import src.kernel.predictors.common as common_module

    common_module._memory_access_templates.clear()
    original_bind = MemoryAccessClassDemand.bind_validated_logical_bytes
    template_ids = []

    def record_template(template, **kwargs):
        template_ids.append(id(template))
        return original_bind(template, **kwargs)

    monkeypatch.setattr(
        MemoryAccessClassDemand,
        "bind_validated_logical_bytes",
        record_template)
    signature = MemoryAccessClassSignature(
        access_class="shared_source_operand",
        dtype="bf16",
        tile_m=16,
        tile_k=64,
        logical_row_bytes=2048,
    )
    c500_demand = bind_memory_access_class_template(
        "shared_source_operand", "l2_cache", 1024.0, signature)
    a800_demand = bind_memory_access_class_template(
        "shared_source_operand", "l2_cache", 2048.0, signature)

    assert c500_demand.logical_bytes == pytest.approx(1024.0)
    assert a800_demand.logical_bytes == pytest.approx(2048.0)
    assert template_ids[0] == template_ids[1]
    assert len(common_module._memory_access_templates) == 1
    common_module._memory_access_templates.clear()


def test_absolute_duration_stage_is_attention_envelope_only():
    inp = _make_input(
        active_units=1,
        grid_size=1,
        include_dispatch_overhead=False,
    )
    inp = dataclasses.replace(
        inp,
        pipeline_plan=PipelineWorkPlan(
            worker_count=1,
            tasks=(PipelineTaskDemand(
                name="worker0",
                worker=0,
                stages=(PipelineStageDemand(
                    "completion", "mechanism:test_completion", None,
                    absolute_duration_s=1.0e-6,
                ),),
            ),),
        ),
    )

    with pytest.raises(
            ValueError, match="only by attention pipeline envelopes"):
        FORMULA.predict_kernel(inp)


def _empty_demand():
    return {
        "l1_cache": 0.0,
        "smem": 0.0,
        "xbar": 0.0,
        "l2_cache": 0.0,
        "dram": 0.0,
        "mte3": 0.0,
        "compute_tensor": 0.0,
        "compute_vector": 0.0,
        "compute_reduction": 0.0,
        "compute_fma": 0.0,
        "compute_sfu": 0.0,
        "compute_alu": 0.0,
        "warp_issue": 0.0,
        "scalar_aiv": 0.0,
    }


def _make_input(**kwargs):
    if "contract_hardware" in kwargs:
        hardware = kwargs.pop("contract_hardware")
    else:
        hardware = HW
    demand = _empty_demand()
    if "demand_per_resource" in kwargs:
        demand.update(kwargs.pop("demand_per_resource"))
    if hardware.kernel_memory_model == "npu_tile":
        support_level = "aggregate_only"
        limitations = (
            "synthetic_test_contract",
            "workload_producer_independence_unavailable",
        )
    else:
        support_level = "partial"
        limitations = ("synthetic_test_contract",)
    if hardware.kernel_memory_model == "npu_tile":
        launch_task_kind = "aiv"
    elif hardware.kernel_memory_model in (
            "gpu_mshr", "pim_bank", "systolic_scratchpad"):
        launch_task_kind = "generic"
    else:
        raise AssertionError(hardware.kernel_memory_model)
    defaults = dict(
        demand_per_resource=demand,
        eta=1.0,
        wave_efficiency=1.0,
        active_units=hardware.num_compute_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=hardware.num_compute_units,
        block_size=hardware.parallel_width,
        variant="test",
        mem_model=hardware.kernel_memory_model,
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind=launch_task_kind,
        include_dispatch_overhead=(hardware.kernel_memory_model != "npu_tile"),
        support_level=support_level,
        limitations=limitations,
    )
    defaults.update(kwargs)
    raw = PredictionInput(**defaults)
    if raw.mem_model == "gpu_mshr":
        transaction_bytes = None
        path_name = "global_memory"
    elif raw.mem_model == "npu_tile":
        transaction_bytes = None
        path_name = "mte2_read"
    elif raw.mem_model in ("pim_bank", "systolic_scratchpad"):
        transaction_bytes = None
        path_name = None
    else:
        raise AssertionError(raw.mem_model)
    if raw.mem_model == "npu_tile":
        independent_requests_per_producer = None
    else:
        independent_requests_per_producer = 1.0
    return attach_structured_contract(
        raw,
        hardware,
        transaction_bytes=transaction_bytes,
        path_name=path_name,
        independent_requests_per_producer=independent_requests_per_producer,
    )


class TestAnalyticalDependencyPath:
    def test_unknown_read_parallelism_keeps_first_request_latency_floor(self):
        base = _make_input(
            demand_per_resource={
                "l2_cache": 32.0,
                "dram": 32.0,
                "compute_alu": 1.0,
            },
            include_dispatch_overhead=False,
        )
        read_path = base.memory_paths[0]
        unknown_source = dataclasses.replace(
            read_path.source,
            independent_requests_per_producer=None,
        )
        path = dataclasses.replace(read_path, source=unknown_source)
        inp = dataclasses.replace(
            base,
            memory_paths=(path,),
            dependency_paths=(AnalyticalDependencyPath(
                "read_then_alu",
                ("memory:global_memory", "compute:compute_alu"),
            ),),
            limitations=base.limitations + (
                "workload_producer_independence_unavailable",),
        )

        prediction = FORMULA.predict_kernel(inp)
        read_diagnostic, = prediction.diagnostics.memory

        assert read_diagnostic.first_request_latency_s is not None
        assert prediction.t_per_group["memory:global_memory"] == (
            pytest.approx(read_diagnostic.first_request_latency_s))

    def test_path_can_price_an_additional_dependent_read_level(self):
        base = _make_input(
            demand_per_resource={
                "l2_cache": 1024.0,
                "dram": 1024.0,
            },
            include_dispatch_overhead=False,
        )
        path = AnalyticalDependencyPath(
            "indexed_read",
            ("memory:global_memory", "memory_latency:global_memory"),
        )

        prediction = FORMULA.predict_kernel(dataclasses.replace(
            base, dependency_paths=(path,)))
        read_diagnostic, = prediction.diagnostics.memory
        expected = math.fsum((
            max(
                read_diagnostic.steady_time_s,
                read_diagnostic.first_request_latency_s,
            ),
            read_diagnostic.first_request_latency_s,
        ))

        assert prediction.t_per_group["dependency:indexed_read"] == (
            pytest.approx(expected))

    def test_path_overlaps_services_within_one_stage(self):
        base = _make_input(
            demand_per_resource={
                "l2_cache": 1024.0,
                "dram": 1024.0,
                "compute_alu": 1_000_000.0,
            },
            include_dispatch_overhead=False,
        )
        path = AnalyticalDependencyPath(
            "load_then_alu",
            ("memory:global_memory", "compute:compute_alu"),
            concurrent_service_groups=((
                "memory:global_memory", "compute:compute_alu"),),
        )

        prediction = FORMULA.predict_kernel(dataclasses.replace(
            base, dependency_paths=(path,)))
        expected = max(
            prediction.t_per_group["memory:global_memory"],
            prediction.diagnostics.compute_time_s["compute_alu"],
        )

        assert prediction.t_per_group["dependency:load_then_alu"] == (
            pytest.approx(expected))

    def test_path_can_repeat_a_source_defined_serial_service(self):
        base = _make_input(
            demand_per_resource={
                "l2_cache": 1024.0,
                "dram": 1024.0,
            },
            include_dispatch_overhead=False,
        )
        path = AnalyticalDependencyPath(
            "three_read_levels",
            ("memory:global_memory", "memory_latency:global_memory"),
            serial_service_counts=(("memory_latency:global_memory", 2),),
        )

        prediction = FORMULA.predict_kernel(dataclasses.replace(
            base, dependency_paths=(path,)))
        read_diagnostic, = prediction.diagnostics.memory
        expected = math.fsum((
            max(
                read_diagnostic.steady_time_s,
                read_diagnostic.first_request_latency_s,
            ),
            2 * read_diagnostic.first_request_latency_s,
        ))

        assert prediction.t_per_group["dependency:three_read_levels"] == (
            pytest.approx(expected))

    def test_path_can_serialize_decomposed_launch_submission(self):
        base = _make_input(
            demand_per_resource={"compute_alu": 1_000_000.0},
            include_dispatch_overhead=False,
        )
        path = AnalyticalDependencyPath(
            "submitted_alu", ("launch:submit", "compute:compute_alu"))

        prediction = FORMULA.predict_kernel(dataclasses.replace(
            base, dependency_paths=(path,)))
        expected = math.fsum((
            HW.launch_latency_model["submit_ns"] * 1e-9,
            prediction.diagnostics.compute_time_s["compute_alu"],
        ))

        assert prediction.t_per_group["dependency:submitted_alu"] == (
            pytest.approx(expected))
        assert prediction.diagnostics.critical_path == (
            "submitted_alu/launch:submit",
            "submitted_alu/compute:compute_alu",
        )

    def test_nonempty_scheduler_path_orders_only_submit_prefix(self):
        base = _make_input(
            contract_hardware=C500_HW,
            demand_per_resource={"compute_alu": 1_000_000.0},
            grid_size=10 * C500_HW.num_compute_units,
            block_size=4 * C500_HW.parallel_width,
            launch_grid=(10 * C500_HW.num_compute_units, 1, 1),
            launch_block=(4 * C500_HW.parallel_width, 1, 1),
        )
        path = AnalyticalDependencyPath(
            "submitted_alu", ("launch:submit", "compute:compute_alu"))

        prediction = PerformanceFormula(C500_HW).predict_kernel(
            dataclasses.replace(base, dependency_paths=(path,)))
        compute_s = prediction.diagnostics.compute_time_s["compute_alu"]
        submit_s = C500_HW.launch_latency_model["submit_ns"] * 1e-9
        complete_task_s = C500_HW.empty_kernel_duration_for_launch(
            base.grid_size,
            base.block_size,
            base.launch_grid,
            base.launch_block,
        ) * 1e-9

        assert prediction.t_per_group["dependency:submitted_alu"] == (
            pytest.approx(submit_s + compute_s))
        assert prediction.t_total_s == pytest.approx(max(
            complete_task_s, submit_s + compute_s))

    def test_path_adds_complete_resource_services_in_declared_order(self):
        base = _make_input(
            demand_per_resource={
                "l2_cache": 1024 * 1024.0,
                "dram": 1024 * 1024.0,
                "compute_alu": 1_000_000.0,
            },
            include_dispatch_overhead=False,
        )
        baseline = FORMULA.predict_kernel(base)
        path = AnalyticalDependencyPath(
            "load_then_alu",
            ("memory:global_memory", "compute:compute_alu"),
        )

        prediction = FORMULA.predict_kernel(dataclasses.replace(
            base, dependency_paths=(path,)))
        expected = math.fsum((
            baseline.t_per_group["memory:global_memory"],
            baseline.diagnostics.compute_time_s["compute_alu"],
        ))

        assert prediction.t_per_group["dependency:load_then_alu"] == (
            pytest.approx(expected))
        assert prediction.t_total_s == pytest.approx(expected)
        assert prediction.bottleneck_group == "dependency:load_then_alu"
        assert prediction.diagnostics.critical_path == (
            "load_then_alu/memory:global_memory",
            "load_then_alu/compute:compute_alu",
        )

    def test_unknown_path_service_fails_loudly(self):
        inp = _make_input(include_dispatch_overhead=False)
        inp = dataclasses.replace(
            inp,
            dependency_paths=(AnalyticalDependencyPath(
                "unknown", ("compute:missing_role",)),),
        )

        with pytest.raises(ValueError, match="unknown services"):
            FORMULA.predict_kernel(inp)

    def test_path_cannot_coexist_with_worker_pipeline(self):
        inp = _make_input(demand_per_resource={"compute_alu": 1.0})
        inp = attach_uniform_pipeline(
            inp,
            HW,
            task_count=1,
            stages=(("compute", "compute:compute_alu", False),),
            buffer_slots=1,
        )

        with pytest.raises(ValueError, match="cannot combine"):
            dataclasses.replace(
                inp,
                dependency_paths=(AnalyticalDependencyPath(
                    "compute", ("compute:compute_alu",)),),
            )


class TestFormulaLaunch:
    def test_empty_demand_uses_complete_empty_task_floor(self):
        inp = _make_input()
        prediction = FORMULA.predict_kernel(inp)

        # The card publishes a measured CTA dispatch cost, so the launch is
        # no longer free.  It is additive against execution rather than a
        # second floor, so with no demand the empty-task floor still sets the
        # total and only t_launch_s changes.
        dispatch_s = HW.cta_dispatch_cost_ns(inp.grid_size) * 1e-9
        assert dispatch_s > 0.0
        assert prediction.t_launch_s == pytest.approx(dispatch_s)
        assert prediction.t_total_s == pytest.approx(
            HW.empty_kernel_duration_for_launch(
                inp.grid_size, inp.block_size) * 1e-9)

    def test_nonempty_scheduler_orders_submit_before_worker_pipeline(self):
        base = _make_input(
            contract_hardware=C500_HW,
            demand_per_resource={"compute_alu": 1_000_000.0},
            grid_size=C500_HW.num_compute_units,
            block_size=4 * C500_HW.parallel_width,
            launch_grid=(C500_HW.num_compute_units, 1, 1),
            launch_block=(4 * C500_HW.parallel_width, 1, 1),
        )
        inp = attach_uniform_pipeline(
            base,
            C500_HW,
            task_count=C500_HW.num_compute_units,
            stages=(("compute", "compute:compute_alu", False),),
            buffer_slots=1,
        )

        prediction = PerformanceFormula(C500_HW).predict_kernel(inp)
        pipeline = prediction.diagnostics.pipeline
        assert pipeline is not None
        submit_s = C500_HW.launch_latency_model["submit_ns"] * 1e-9
        complete_task_s = C500_HW.empty_kernel_duration_for_launch(
            inp.grid_size,
            inp.block_size,
            inp.launch_grid,
            inp.launch_block,
        ) * 1e-9
        assert prediction.t_launch_s == pytest.approx(submit_s)
        assert prediction.t_total_s == pytest.approx(max(
            complete_task_s, submit_s + pipeline.execution_time_s))
        assert prediction.bottleneck_group == "launch"

    def test_empty_scheduler_orders_submit_before_worker_pipeline(self):
        base = _make_input(
            demand_per_resource={"compute_alu": 1_000_000.0},
            grid_size=HW.num_compute_units,
            block_size=4 * HW.parallel_width,
            launch_grid=(HW.num_compute_units, 1, 1),
            launch_block=(4 * HW.parallel_width, 1, 1),
        )
        inp = attach_uniform_pipeline(
            base,
            HW,
            task_count=HW.num_compute_units,
            stages=(("compute", "compute:compute_alu", False),),
            buffer_slots=1,
        )

        prediction = FORMULA.predict_kernel(inp)
        pipeline = prediction.diagnostics.pipeline
        assert pipeline is not None
        submit_s = HW.launch_latency_model["submit_ns"] * 1e-9
        complete_task_s = HW.empty_kernel_duration_for_launch(
            inp.grid_size,
            inp.block_size,
            inp.launch_grid,
            inp.launch_block,
        ) * 1e-9
        # Submission is ordered before the worker pipeline, and the launch
        # also pays the card's measured CTA dispatch cost.
        assert prediction.t_launch_s == pytest.approx(
            submit_s + HW.cta_dispatch_cost_ns(inp.grid_size) * 1e-9)
        assert prediction.t_total_s == pytest.approx(max(
            complete_task_s,
            submit_s + HW.cta_dispatch_cost_ns(inp.grid_size) * 1e-9
            + pipeline.execution_time_s))

    def test_empty_floor_overlaps_resource_execution(self):
        inp = _make_input(
            grid_size=16384,
            block_size=1024,
            demand_per_resource={"compute_alu": 1e9},
        )
        prediction = FORMULA.predict_kernel(inp)
        execution = max(prediction.t_per_group.values())

        assert execution > HW.empty_kernel_duration_for_launch(
            inp.grid_size, inp.block_size) * 1e-9
        assert prediction.t_total_s == pytest.approx(
            execution + HW.cta_dispatch_cost_ns(inp.grid_size) * 1e-9)

    def test_typed_memory_dependency_floor_overlaps_empty_and_execution(self):
        contract = "warm_default_cache_memory_dependency_task_v1"
        inp = _make_input(
            grid_size=3,
            block_size=128,
            launch_grid=(3, 1, 1),
            launch_block=(128, 1, 1),
            task_duration_floor_contract=contract,
        )

        prediction = FORMULA.predict_kernel(inp)
        expected_ns = HW.memory_dependency_task_duration_ns(
            contract, 3, 128, (3, 1, 1), (128, 1, 1), inp.sm_freq_hz)

        assert prediction.t_launch_s == 0.0
        assert prediction.t_total_ns == pytest.approx(expected_ns)
        assert prediction.bottleneck_group == "launch"

        execution_bound = _make_input(
            grid_size=3,
            block_size=128,
            launch_grid=(3, 1, 1),
            launch_block=(128, 1, 1),
            task_duration_floor_contract=contract,
            demand_per_resource={"compute_alu": 1e9},
        )
        execution_prediction = FORMULA.predict_kernel(execution_bound)
        assert execution_prediction.t_total_ns > expected_ns
        assert execution_prediction.bottleneck_group != "launch"

    def test_launch_terms_can_be_disabled_explicitly(self):
        inp = _make_input(demand_per_resource={
            "l2_cache": 128.0, "dram": 128.0})
        inp = dataclasses.replace(inp, include_dispatch_overhead=False)
        prediction = FORMULA.predict_kernel(inp)

        assert prediction.t_launch_s == 0.0
        assert prediction.bottleneck_group != "launch"

    def test_independent_additive_dispatch_is_not_a_second_floor(
            self, monkeypatch):
        monkeypatch.setattr(HW, "dispatch_overhead_ns", 1000.0)
        monkeypatch.setattr(HW, "empty_kernel_duration_lut", None)
        monkeypatch.setattr(HW, "launch_latency_model", None)
        inp = _make_input(demand_per_resource={
            "l2_cache": 128.0, "dram": 128.0})
        prediction = FORMULA.predict_kernel(inp)

        # Both the generic dispatch overhead and the measured CTA dispatch
        # cost are additive against execution; neither acts as a floor.
        assert prediction.t_launch_s == pytest.approx(
            1e-6 + HW.cta_dispatch_cost_ns(inp.grid_size) * 1e-9)
        assert prediction.t_total_s >= prediction.t_launch_s


class TestFormulaComputeCapacity:
    @pytest.mark.parametrize(
        ("hardware", "role", "task_kind", "domain_units"),
        (
            (HW, "compute_alu", "generic", 108),
            (C500_HW, "compute_alu", "generic", 104),
            (NPU_HW, "compute_tensor", "aic", 24),
            (NPU_HW, "compute_vector", "aiv", 48),
        ),
    )
    def test_shared_compute_capacity_uses_active_execution_domain_fraction(
            self, hardware, role, task_kind, domain_units):
        formula = PerformanceFormula(hardware)
        operations = hardware.resource_by_role(role).capacity
        common = {
            "contract_hardware": hardware,
            "demand_per_resource": {role: operations},
            "launch_task_kind": task_kind,
            "include_dispatch_overhead": False,
        }
        one = formula.predict_kernel(_make_input(
            **common, active_units=1, grid_size=1))
        full = formula.predict_kernel(_make_input(
            **common, active_units=domain_units, grid_size=domain_units))
        resource_name = hardware.resource_by_role(role).name

        assert full.t_per_resource[resource_name] == pytest.approx(1.0)
        assert one.t_per_resource[resource_name] == pytest.approx(
            float(domain_units))

    def test_shared_memory_capacity_is_not_reduced_with_active_compute_units(
            self):
        byte_count = 1_000_000.0
        one = FORMULA.predict_kernel(_make_input(
            active_units=1,
            grid_size=1,
            demand_per_resource={"l2_cache": byte_count},
            include_dispatch_overhead=False,
        ))
        full = FORMULA.predict_kernel(_make_input(
            active_units=HW.num_compute_units,
            grid_size=HW.num_compute_units,
            demand_per_resource={"l2_cache": byte_count},
            include_dispatch_overhead=False,
        ))
        l2_name = HW.resource_by_role("l2_cache").name

        assert one.t_per_resource[l2_name] == pytest.approx(
            full.t_per_resource[l2_name])

    def test_typed_services_serialize_on_one_physical_compute_resource(
            self, monkeypatch):
        vector = NPU_HW.resource_by_role("compute_vector")
        monkeypatch.setattr(
            vector,
            "service_rates_per_unit_s",
            {
                "generic": 10e9,
                "exp": 2e9,
                "div": 1e9,
            },
        )
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 80.0},
            launch_task_kind="aiv",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        aggregate = inp.compute_demands[0]
        typed = dataclasses.replace(
            aggregate,
            service_demands=(
                ComputeServiceDemand("generic", 40.0),
                ComputeServiceDemand("exp", 20.0),
                ComputeServiceDemand("div", 20.0),
            ),
        )
        prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
            inp, compute_demands=(typed,)))

        expected = 40.0 / (2 * 10e9)
        expected += 20.0 / (2 * 2e9)
        expected += 20.0 / (2 * 1e9)
        assert prediction.t_per_resource[vector.name] == pytest.approx(
            expected)

    def test_untyped_compute_preserves_historical_float_association(self):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 1.5344799230657985e185},
            launch_task_kind="aiv",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        aggregate = dataclasses.replace(
            inp.compute_demands[0],
            shape_efficiency=0.056796553053849785,
            schedule_efficiency=0.6556822842792598,
        )
        inp = dataclasses.replace(inp, compute_demands=(aggregate,))
        vector = NPU_HW.resource_by_role("compute_vector")
        capacity = NPU_FORMULA._compute_capacity(inp, vector)
        expected = aggregate.operations / (
            capacity * aggregate.shape_efficiency
            * aggregate.schedule_efficiency)

        prediction = NPU_FORMULA.predict_kernel(inp)

        assert prediction.t_per_resource[vector.name] == expected

    def test_typed_service_serial_sum_is_order_independent(self, monkeypatch):
        vector = NPU_HW.resource_by_role("compute_vector")
        monkeypatch.setattr(
            vector,
            "service_rates_per_unit_s",
            {
                "generic": 1.0 / 48.0,
                "exp": 1.0 / 48.0,
                "div": 1.0 / 48.0,
            },
        )
        operations = math.fsum((1e16, 1.0, 1.0))
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": operations},
            launch_task_kind="aiv",
            active_units=48,
            grid_size=48,
            include_dispatch_overhead=False,
        )
        aggregate = inp.compute_demands[0]
        services = (
            ComputeServiceDemand("generic", 1e16),
            ComputeServiceDemand("exp", 1.0),
            ComputeServiceDemand("div", 1.0),
        )
        forward = dataclasses.replace(
            aggregate, service_demands=services)
        reverse = dataclasses.replace(
            aggregate, service_demands=tuple(reversed(services)))

        forward_prediction = NPU_FORMULA.predict_kernel(
            dataclasses.replace(inp, compute_demands=(forward,)))
        reverse_prediction = NPU_FORMULA.predict_kernel(
            dataclasses.replace(inp, compute_demands=(reverse,)))

        assert forward_prediction.t_per_resource[vector.name] == operations
        assert reverse_prediction.t_per_resource[vector.name] == operations

    def test_typed_service_requires_an_explicit_hardware_rate(
            self, monkeypatch):
        vector = NPU_HW.resource_by_role("compute_vector")
        monkeypatch.setattr(vector, "service_rates_per_unit_s", None)
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )
        aggregate = inp.compute_demands[0]
        typed = dataclasses.replace(
            aggregate,
            service_demands=(ComputeServiceDemand("exp", 2.0),),
        )

        with pytest.raises(ValueError, match="has no typed service rates"):
            NPU_FORMULA.predict_kernel(dataclasses.replace(
                inp, compute_demands=(typed,)))

    def test_typed_service_adds_critical_instance_setup_without_efficiency(
            self, monkeypatch):
        vector = NPU_HW.resource_by_role("compute_vector")
        monkeypatch.setattr(
            vector, "service_rates_per_unit_s", {"exp": 2e9})
        monkeypatch.setattr(
            vector, "service_setup_time_s", {"exp": 5e-9})
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 20.0},
            launch_task_kind="aiv",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            shape_efficiency=0.5,
            schedule_efficiency=0.25,
            service_demands=(
                ComputeServiceDemand(
                    "exp", 20.0, serial_invocations=3),),
        )

        prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
            inp, compute_demands=(typed,)))

        steady = 20.0 / (2 * 2e9 * 0.5 * 0.25)
        assert prediction.t_per_resource[vector.name] == pytest.approx(
            steady + 3 * 5e-9)

    def test_worker_typed_services_price_the_complete_critical_aiv(
            self, monkeypatch):
        vector = NPU_HW.resource_by_role("compute_vector")
        monkeypatch.setattr(
            vector, "service_rates_per_unit_s", {
                "generic": 10.0,
                "exp": 2.0,
                "div": 1.0,
            })
        monkeypatch.setattr(
            vector, "service_setup_time_s", {
                "exp": 5.0,
                "div": 7.0,
            })
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 44.0},
            launch_task_kind="mixed",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        services = (
            ComputeServiceDemand(
                "generic", 30.0,
                operations_by_worker=(10.0, 0.0, 20.0, 0.0),
                serial_invocations_by_worker=(0, 0, 0, 0),
            ),
            ComputeServiceDemand(
                "exp", 6.0, serial_invocations=2,
                operations_by_worker=(2.0, 0.0, 4.0, 0.0),
                serial_invocations_by_worker=(1, 0, 2, 0),
            ),
            ComputeServiceDemand(
                "div", 8.0, serial_invocations=1,
                operations_by_worker=(0.0, 0.0, 8.0, 0.0),
                serial_invocations_by_worker=(0, 0, 1, 0),
            ),
        )
        typed = dataclasses.replace(
            inp.compute_demands[0], service_demands=services)

        prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
            inp, compute_demands=(typed,)))

        critical_worker = (
            20.0 / 10.0 + 4.0 / 2.0 + 2 * 5.0
            + 8.0 / 1.0 + 7.0)
        assert prediction.t_per_resource[vector.name] == pytest.approx(
            critical_worker)

    def test_worker_typed_services_require_physical_efficiencies(self):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        service = ComputeServiceDemand(
            "generic", 2.0,
            operations_by_worker=(1.0, 1.0),
            serial_invocations_by_worker=(0, 0),
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            schedule_efficiency=0.5,
            service_demands=(service,),
        )

        with pytest.raises(
                ValueError, match="physical operations and unit efficiencies"):
            NPU_FORMULA.predict_kernel(dataclasses.replace(
                inp, compute_demands=(typed,)))

    def test_exact_request_service_uses_critical_worker_without_efficiency(
            self):
        vector = NPU_HW.resource_by_role("compute_vector")
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 31.0},
            launch_task_kind="aiv",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        exact = ComputeExactRequestDemand(
            "norm_sqrt_duplicate_div_serialized",
            ((1, 8), (2,)),
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            shape_efficiency=0.5,
            schedule_efficiency=0.25,
            service_demands=(ComputeServiceDemand("generic", 20.0),),
            exact_request_demands=(exact,),
        )

        prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
            inp, compute_demands=(typed,)))

        generic = 20.0 / (2 * 230.4e9 * 0.5 * 0.25)
        critical_worker = 55.555826823e-9 + 45.555826823e-9
        assert prediction.t_per_resource[vector.name] == pytest.approx(
            generic + critical_worker)

    def test_pipelined_exact_requests_keep_work_but_use_completion_envelope(
            self):
        vector = NPU_HW.resource_by_role("compute_vector")
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 31.0},
            launch_task_kind="aiv",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        exact = ComputeExactRequestDemand(
            "norm_sqrt_duplicate_div_serialized",
            ((1, 8), (2,)),
            pipelined_completion=True,
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(ComputeServiceDemand("generic", 20.0),),
            exact_request_demands=(exact,),
        )

        prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
            inp, compute_demands=(typed,)))

        generic = 20.0 / (2 * 230.4e9)
        completion_envelope = max(
            vector.exact_request_service_time_s(
                "norm_sqrt_duplicate_div_serialized", request_size)
            for request_size in (1, 2, 8)
        )
        assert exact.operations == 11.0
        assert prediction.t_per_resource[vector.name] == pytest.approx(
            generic + completion_envelope)

    def test_worker_typed_and_exact_services_share_one_critical_worker(self):
        vector = NPU_HW.resource_by_role("compute_vector")
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 31.0},
            launch_task_kind="aiv",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(ComputeServiceDemand(
                "generic", 20.0,
                operations_by_worker=(1.0, 19.0),
                serial_invocations_by_worker=(0, 0),
            ),),
            exact_request_demands=(ComputeExactRequestDemand(
                "norm_sqrt_duplicate_div_serialized",
                ((1, 8), (2,)),
            ),),
        )

        prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
            inp, compute_demands=(typed,)))

        worker0 = (
            1.0 / 230.4e9 + 55.555826823e-9 + 45.555826823e-9)
        worker1 = 19.0 / 230.4e9 + 55.555826823e-9
        assert prediction.t_per_resource[vector.name] == pytest.approx(
            max(worker0, worker1))

    def test_mixed_exact_service_uses_physical_aiv_worker_count(self):
        vector = NPU_HW.resource_by_role("compute_vector")
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 4.0},
            launch_task_kind="mixed",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        exact = dataclasses.replace(
            inp.compute_demands[0],
            exact_request_demands=(ComputeExactRequestDemand(
                "norm_sqrt_duplicate_div_serialized",
                ((1,), (1,), (1,), (1,)),
            ),),
        )

        prediction = NPU_FORMULA.predict_kernel(dataclasses.replace(
            inp, compute_demands=(exact,)))

        assert prediction.t_per_resource[vector.name] == pytest.approx(
            55.555826823e-9)

    def test_exact_service_rejects_fractional_physical_worker_ratio(
            self, monkeypatch):
        vector = NPU_HW.resource_by_role("compute_vector")
        monkeypatch.setattr(vector, "total_units", 47)
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 4.0},
            launch_task_kind="mixed",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        exact = dataclasses.replace(
            inp.compute_demands[0],
            exact_request_demands=(ComputeExactRequestDemand(
                "norm_sqrt_duplicate_div_serialized",
                ((1,), (1,), (1,), (1,)),
            ),),
        )

        with pytest.raises(ValueError, match="worker count is fractional"):
            NPU_FORMULA.predict_kernel(dataclasses.replace(
                inp, compute_demands=(exact,)))

    def test_exact_request_service_fails_on_worker_or_size_mismatch(
            self, monkeypatch):
        vector = NPU_HW.resource_by_role("compute_vector")
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            active_units=2,
            grid_size=2,
            include_dispatch_overhead=False,
        )
        one_worker = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(ComputeServiceDemand("generic", 1.0),),
            exact_request_demands=(ComputeExactRequestDemand(
                "norm_sqrt_duplicate_div_serialized", ((1,),)),),
        )
        with pytest.raises(ValueError, match="active physical units"):
            NPU_FORMULA.predict_kernel(dataclasses.replace(
                inp, compute_demands=(one_worker,)))

        exact_services = vector.exact_request_services
        monkeypatch.setattr(vector, "exact_request_services", {
            "norm_sqrt_duplicate_div_serialized": {
                **exact_services["norm_sqrt_duplicate_div_serialized"],
                "service_time_s_by_request_size": {1: 10e-9},
            },
        })
        missing_size = dataclasses.replace(
            inp.compute_demands[0],
            exact_request_demands=(ComputeExactRequestDemand(
                "norm_sqrt_duplicate_div_serialized", ((2,), ())),),
        )
        with pytest.raises(ValueError, match="has no request size 2"):
            NPU_FORMULA.predict_kernel(dataclasses.replace(
                inp, compute_demands=(missing_size,)))

    def test_worker_and_exact_schedules_fail_at_pipeline_input_boundary(self):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )
        pipeline = attach_uniform_pipeline(
            dataclasses.replace(inp, mem_model="gpu_mshr"), NPU_HW,
            task_count=1,
            stages=(("compute", "compute:compute_vector", False),),
            buffer_slots=1,
        )
        exact = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(ComputeServiceDemand("generic", 1.0),),
            exact_request_demands=(ComputeExactRequestDemand(
                "norm_sqrt_duplicate_div_serialized", ((1,),)),),
        )
        with pytest.raises(ValueError, match="exact request services"):
            dataclasses.replace(pipeline, compute_demands=(exact,))

        worker = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(ComputeServiceDemand(
                "generic", 2.0,
                operations_by_worker=(2.0,),
                serial_invocations_by_worker=(0,),
            ),),
        )
        with pytest.raises(ValueError, match="worker-scheduled typed"):
            dataclasses.replace(pipeline, compute_demands=(worker,))

    def test_positive_serial_invocations_require_service_setup(
            self, monkeypatch):
        vector = NPU_HW.resource_by_role("compute_vector")
        monkeypatch.setattr(
            vector, "service_rates_per_unit_s", {"exp": 2e9})
        monkeypatch.setattr(vector, "service_setup_time_s", None)
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(
                ComputeServiceDemand(
                    "exp", 2.0, serial_invocations=1),),
        )

        with pytest.raises(ValueError, match="no typed service setup times"):
            NPU_FORMULA.predict_kernel(dataclasses.replace(
                inp, compute_demands=(typed,)))

    def test_typed_service_requires_its_own_hardware_rate(self, monkeypatch):
        vector = NPU_HW.resource_by_role("compute_vector")
        monkeypatch.setattr(
            vector, "service_rates_per_unit_s", {"generic": 10e9})
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(ComputeServiceDemand("exp", 2.0),),
        )

        with pytest.raises(ValueError, match="has no typed service 'exp'"):
            NPU_FORMULA.predict_kernel(dataclasses.replace(
                inp, compute_demands=(typed,)))

    def test_typed_service_demand_must_conserve_aggregate_operations(self):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )

        with pytest.raises(ValueError, match="must conserve"):
            dataclasses.replace(
                inp.compute_demands[0],
                service_demands=(ComputeServiceDemand("exp", 1.0),),
            )

        tiny = dataclasses.replace(
            inp.compute_demands[0], operations=1e-12)
        with pytest.raises(ValueError, match="must conserve"):
            dataclasses.replace(
                tiny,
                service_demands=(ComputeServiceDemand("exp", 2e-12),),
            )

        large = dataclasses.replace(
            inp.compute_demands[0], operations=1e24)
        with pytest.raises(ValueError, match="must conserve"):
            dataclasses.replace(
                large,
                service_demands=(
                    ComputeServiceDemand("generic", 1e24 + 1e12),),
            )

        zero = dataclasses.replace(
            inp.compute_demands[0], operations=0.0)
        with pytest.raises(ValueError, match="positive aggregate"):
            dataclasses.replace(
                zero,
                service_demands=(
                    ComputeServiceDemand("generic", math.ulp(0.0)),),
            )

        with pytest.raises(TypeError, match="serial_invocations.*integer"):
            ComputeServiceDemand("exp", 1.0, serial_invocations=1.5)
        with pytest.raises(ValueError, match="cannot be negative"):
            ComputeServiceDemand("exp", 1.0, serial_invocations=-1)

    def test_worker_typed_service_conservation_is_scale_relative(self):
        ComputeServiceDemand(
            "generic", 1e-200,
            operations_by_worker=(0.5e-200, 0.5e-200),
            serial_invocations_by_worker=(0, 0),
        )

        with pytest.raises(ValueError, match="workers=.*aggregate"):
            ComputeServiceDemand(
                "generic", 2e-200,
                operations_by_worker=(0.5e-200, 0.5e-200),
                serial_invocations_by_worker=(0, 0),
            )

    def test_typed_services_reject_non_vector_roles_and_empirical_labels(self):
        inp = _make_input(demand_per_resource={"compute_fma": 1.0})

        with pytest.raises(ValueError, match="supported only.*compute_vector"):
            dataclasses.replace(
                inp.compute_demands[0],
                service_demands=(ComputeServiceDemand("generic", 1.0),),
            )
        with pytest.raises(ValueError, match="must be one of"):
            ComputeServiceDemand("activation_shape_1024", 1.0)
        with pytest.raises(ValueError, match="must be one of"):
            ComputeExactRequestDemand("norm_shape_1024", ((1,),))
        with pytest.raises(TypeError, match="worker requests must be tuples"):
            ComputeExactRequestDemand(
                "norm_sqrt_duplicate_div_serialized", ([1],))

        exact = ComputeExactRequestDemand(
            "norm_sqrt_duplicate_div_serialized", ((1,),))
        with pytest.raises(
                ValueError, match="unsupported on compute role.*compute_tensor"):
            dataclasses.replace(
                inp.compute_demands[0], resource_role="compute_tensor",
                exact_request_demands=(exact,))

    def test_structured_replacement_cannot_rescale_exact_request_schedule(self):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )
        exact = dataclasses.replace(
            inp.compute_demands[0],
            exact_request_demands=(ComputeExactRequestDemand(
                "norm_sqrt_duplicate_div_serialized", ((2,),)),),
        )
        inp = dataclasses.replace(inp, compute_demands=(exact,))

        with pytest.raises(ValueError, match="cannot rescale"):
            replace_structured_demands(inp, {"compute_vector": 4.0})

    def test_structured_replacement_preserves_exact_request_schedule(self):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 10.0},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )
        exact = ComputeExactRequestDemand(
            "norm_sqrt_duplicate_div_serialized", ((2,),))
        typed = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(ComputeServiceDemand("generic", 8.0),),
            exact_request_demands=(exact,),
        )
        inp = dataclasses.replace(inp, compute_demands=(typed,))

        replaced = replace_structured_demands(
            inp, {"compute_vector": 6.0})

        demand, = replaced.compute_demands
        assert demand.operations == 6.0
        assert demand.service_demands == (
            ComputeServiceDemand("generic", 4.0),)
        assert demand.exact_request_demands == (exact,)

    def test_typed_services_reject_duplicate_physical_services(self):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )

        with pytest.raises(ValueError, match="cannot repeat services"):
            dataclasses.replace(
                inp.compute_demands[0],
                service_demands=(
                    ComputeServiceDemand("generic", 1.0),
                    ComputeServiceDemand("generic", 1.0),
                ),
            )

    def test_structured_replacement_scales_typed_services_proportionally(self):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 8.0},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(
                ComputeServiceDemand("generic", 6.0),
                ComputeServiceDemand("exp", 1.0),
                ComputeServiceDemand("div", 1.0),
            ),
        )
        inp = dataclasses.replace(inp, compute_demands=(typed,))

        replaced = replace_structured_demands(
            inp, {"compute_vector": 5.0})

        assert replaced.compute_demands[0].operations == 5.0
        assert tuple(
            (service.service, service.operations)
            for service in replaced.compute_demands[0].service_demands
        ) == (("generic", 3.75), ("exp", 0.625), ("div", 0.625))

    @pytest.mark.parametrize(
        ("prior_operations", "replacement_operations"),
        ((1e200, 1e200), (1e-200, 1e200), (1e200, 1e-200)),
    )
    def test_structured_replacement_scales_typed_services_without_overflow(
            self, prior_operations, replacement_operations):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": prior_operations},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(
                ComputeServiceDemand("generic", prior_operations),),
        )
        inp = dataclasses.replace(inp, compute_demands=(typed,))

        replaced = replace_structured_demands(
            inp, {"compute_vector": replacement_operations})

        assert replaced.compute_demands[0].operations == replacement_operations
        assert replaced.compute_demands[0].service_demands == (
            ComputeServiceDemand("generic", replacement_operations),)

    @pytest.mark.parametrize("operations", (1e200, 1e-200))
    def test_structured_replacement_preserves_multiple_extreme_services(
            self, operations):
        half_operations = operations / 2.0
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": operations},
            launch_task_kind="aiv",
            active_units=1,
            grid_size=1,
            include_dispatch_overhead=False,
        )
        typed = dataclasses.replace(
            inp.compute_demands[0],
            service_demands=(
                ComputeServiceDemand("generic", half_operations),
                ComputeServiceDemand("exp", half_operations),
            ),
        )
        inp = dataclasses.replace(inp, compute_demands=(typed,))

        replaced = replace_structured_demands(
            inp, {"compute_vector": operations})

        assert replaced.compute_demands[0].service_demands == (
            ComputeServiceDemand("generic", half_operations),
            ComputeServiceDemand("exp", half_operations),
        )


class TestFormulaMemory:
    @staticmethod
    def _local_source_input(worker_count):
        transaction_bytes = float(HW.l2_transaction_bytes)
        base = _make_input(
            active_units=worker_count,
            grid_size=worker_count,
            demand_per_resource={
                "l2_cache": worker_count * transaction_bytes,
                "dram": transaction_bytes,
            },
            include_dispatch_overhead=False,
        )
        stage = PipelineStageDemand(
            "load", "memory:global_memory", None,
            memory_node_demands=(PhaseMemoryDemand(
                "global_memory", "l2_cache", transaction_bytes),),
            launch_shared_memory_node_demands=(PhaseMemoryDemand(
                "global_memory", "dram", transaction_bytes),),
            local_memory_source_completion="sequence_start",
        )
        plan = PipelineWorkPlan(
            worker_count=worker_count,
            tasks=tuple(
                PipelineTaskDemand(
                    name=f"worker{worker}", worker=worker,
                    stages=(stage,))
                for worker in range(worker_count)),
        )
        return dataclasses.replace(base, pipeline_plan=plan)

    def test_local_source_completion_is_invariant_to_shared_path_amortization(
            self):
        one = FORMULA.predict_kernel(self._local_source_input(1))
        two = FORMULA.predict_kernel(self._local_source_input(2))

        one_memory, = one.diagnostics.memory
        two_memory, = two.diagnostics.memory
        expected = math.fsum(
            node.unloaded_latency_s for node in one_memory.nodes)
        one_stage = one.diagnostics.pipeline.tasks[0].stage_end_s["load"]
        two_stage = two.diagnostics.pipeline.tasks[0].stage_end_s["load"]

        assert one_stage == pytest.approx(expected)
        assert two_stage == pytest.approx(expected)
        assert two_memory.first_request_latency_s < expected
        assert two.t_total_s >= two.t_per_group["memory:global_memory"]

    def test_local_source_completion_fails_without_transaction_geometry(self):
        base = _make_input(
            contract_hardware=NPU_HW,
            active_units=1,
            grid_size=1,
            demand_per_resource={"l2_cache": 1024.0, "dram": 1024.0},
            include_dispatch_overhead=False,
        )
        path, = base.memory_paths
        stage = PipelineStageDemand(
            "load", "memory:mte2_read", None,
            memory_node_demands=tuple(
                PhaseMemoryDemand(path.name, node.resource_role, node.bytes)
                for node in path.nodes),
            local_memory_source_completion="sequence_start",
        )
        malformed = dataclasses.replace(
            base,
            pipeline_plan=PipelineWorkPlan(
                worker_count=1,
                tasks=(PipelineTaskDemand(
                    name="worker0", worker=0, stages=(stage,)),),
            ),
            support_level="partial",
        )

        with pytest.raises(
                ValueError, match="requires a transaction-resolved"):
            NPU_FORMULA.predict_kernel(malformed)

    def test_large_dram_demand_reports_the_binding_physical_constraint(self):
        traffic = 100 * 1024 ** 3
        inp = _make_input(demand_per_resource={
            "l2_cache": traffic, "dram": traffic})
        prediction = FORMULA.predict_kernel(inp)
        memory = prediction.diagnostics.memory[0]

        assert memory.bottleneck == "source"
        assert memory.lambda_initial_per_s < memory.lambda_capacity_per_s
        assert 0.0 < memory.nodes[-1].rho < 1.0
        assert memory.nodes[-1].queue_wait_s == 0.0

    def test_memory_time_increases_with_transaction_count(self):
        small = FORMULA.predict_kernel(_make_input(
            demand_per_resource={"l2_cache": 1e9, "dram": 1e9}))
        large = FORMULA.predict_kernel(_make_input(
            demand_per_resource={"l2_cache": 10e9, "dram": 10e9}))

        assert large.t_total_s > small.t_total_s

    def test_flat_and_structured_memory_must_remain_consistent(self):
        inp = _make_input(demand_per_resource={
            "l2_cache": 1024.0, "dram": 512.0})
        broken = dataclasses.replace(
            inp,
            demand_per_resource={
                **inp.demand_per_resource,
                "dram": 256.0,
            },
        )

        with pytest.raises(ValueError, match="memory demand amounts disagree"):
            FORMULA.predict_kernel(broken)

    def test_flat_memory_cannot_introduce_an_unstructured_downstream_node(self):
        inp = _make_input(demand_per_resource={"l2_cache": 1024.0})
        broken = dataclasses.replace(
            inp,
            demand_per_resource={
                **inp.demand_per_resource,
                "dram": 128.0,
            },
        )

        with pytest.raises(ValueError, match="memory demand amounts disagree"):
            FORMULA.predict_kernel(broken)

    def test_canonical_replacement_updates_structured_transactions(self):
        inp = _make_input(demand_per_resource={
            "l2_cache": 1024.0, "dram": 512.0})
        replaced = replace_structured_demands(
            inp, {"l2_cache": 2048.0, "dram": 1024.0})
        prediction = FORMULA.predict_kernel(replaced)

        diagnostic = prediction.diagnostics.memory[0]
        assert diagnostic.nodes[0].bytes == 2048.0
        assert diagnostic.nodes[1].bytes == 1024.0

    def test_short_tile_pays_first_request_then_remaining_transactions(self):
        inp = _make_input(
            demand_per_resource={
                "l2_cache": 256 * 1024.0,
                "compute_alu": 1.0,
            },
            include_dispatch_overhead=False,
        )
        inp = attach_uniform_pipeline(
            inp,
            HW,
            task_count=1,
            stages=(
                ("load", "memory:global_memory", True),
                ("compute", "compute:compute_alu", False),
            ),
            buffer_slots=1,
        )
        prediction = FORMULA.predict_kernel(inp)
        memory = prediction.diagnostics.memory[0]
        pipeline = prediction.diagnostics.pipeline
        # A stage that does not declare a single request burst keeps streaming
        # its remaining transactions against the launch's own solved rate.
        expected = (
            memory.first_request_latency_s
            + (memory.initial_transactions - 1.0)
            / memory.lambda_initial_per_s)

        assert pipeline is not None
        assert pipeline.tasks[0].stage_end_s["load"] == pytest.approx(expected)
        assert pipeline.tasks[0].stage_end_s["load"] > memory.steady_time_s

    def test_each_worker_single_request_tile_pays_only_first_latency(self):
        inp = _make_input(
            active_units=2,
            grid_size=2,
            demand_per_resource={
                "l2_cache": 2.0 * HW.l2_transaction_bytes,
                "compute_alu": 2.0,
            },
            include_dispatch_overhead=False,
        )
        inp = attach_uniform_pipeline(
            inp,
            HW,
            task_count=2,
            stages=(
                ("load", "memory:global_memory", True),
                ("compute", "compute:compute_alu", False),
            ),
            buffer_slots=1,
        )

        prediction = FORMULA.predict_kernel(inp)
        memory = prediction.diagnostics.memory[0]
        pipeline = prediction.diagnostics.pipeline

        assert pipeline is not None
        assert memory.initial_transactions == 2.0
        assert pipeline.tasks[0].stage_end_s["load"] == pytest.approx(
            memory.first_request_latency_s)
        assert pipeline.tasks[1].stage_end_s["load"] == pytest.approx(
            memory.first_request_latency_s)

    def test_directional_dram_is_priced_once_as_a_joint_half_duplex_bound(self):
        read_bytes = 3_000_000.0
        write_bytes = 2_000_000.0
        inp = _make_input(
            demand_per_resource={
                "l2_cache": read_bytes + write_bytes,
                "dram": read_bytes + write_bytes,
            },
            include_dispatch_overhead=False,
        )
        inp = split_global_memory_paths(
            inp,
            read_l2_bytes=read_bytes,
            read_dram_bytes=read_bytes,
            write_l2_bytes=write_bytes,
            write_dram_bytes=write_bytes,
        )

        prediction = FORMULA.predict_kernel(inp)
        arbitration = prediction.diagnostics.memory_arbitration[0]
        read_capacity = HW.dram_capacity_for_access(
            "read",
            active_units=inp.active_units,
            sm_freq_hz=inp.sm_freq_hz,
            dram_freq_hz=inp.dram_freq_hz,
        )
        write_capacity = HW.dram_capacity_for_access(
            "write",
            active_units=inp.active_units,
            sm_freq_hz=inp.sm_freq_hz,
            dram_freq_hz=inp.dram_freq_hz,
        )
        expected = read_bytes / read_capacity + write_bytes / write_capacity
        dram_resource = HW.resource_by_role("dram").name

        assert prediction.t_per_resource[dram_resource] == pytest.approx(
            expected)
        assert arbitration.analytical_time_s == pytest.approx(expected)
        assert arbitration.read_capacity_per_s == pytest.approx(read_capacity)
        assert arbitration.write_capacity_per_s == pytest.approx(
            write_capacity)
        assert arbitration.switch_count is None
        assert arbitration.schedule_status == "unavailable"

    def test_dram_override_marks_joint_diagnostic_as_overridden(self):
        byte_count = 1_000_000.0
        inp = _make_input(
            demand_per_resource={
                "l2_cache": byte_count,
                "dram": byte_count,
            },
            include_dispatch_overhead=False,
            resource_time_overrides_ns={"dram": 123.0},
        )

        prediction = FORMULA.predict_kernel(inp)
        arbitration = prediction.diagnostics.memory_arbitration[0]
        dram_resource = HW.resource_by_role("dram").name

        assert prediction.t_per_resource[dram_resource] == pytest.approx(
            123e-9)
        assert arbitration.overridden
        assert arbitration.effective_time_s == pytest.approx(123e-9)
        assert arbitration.analytical_time_s != pytest.approx(123e-9)

    def test_dram_override_propagates_to_pipeline_memory_service(self):
        byte_count = 1_000_000.0
        override_ns = 123_000.0
        inp = _make_input(
            demand_per_resource={
                "l2_cache": byte_count,
                "dram": byte_count,
            },
            include_dispatch_overhead=False,
            resource_time_overrides_ns={"dram": override_ns},
        )
        inp = attach_uniform_pipeline(
            inp,
            HW,
            task_count=1,
            stages=(("load", "memory:global_memory", True),),
            buffer_slots=1,
        )

        prediction = FORMULA.predict_kernel(inp)
        pipeline = prediction.diagnostics.pipeline

        assert pipeline is not None
        assert pipeline.execution_time_s == pytest.approx(override_ns * 1e-9)
        assert prediction.t_total_s == pytest.approx(override_ns * 1e-9)

    def test_aggregate_dram_override_rejects_multiple_memory_paths(self):
        read_bytes = 3_000_000.0
        write_bytes = 2_000_000.0
        inp = _make_input(
            demand_per_resource={
                "l2_cache": read_bytes + write_bytes,
                "dram": read_bytes + write_bytes,
            },
            include_dispatch_overhead=False,
            resource_time_overrides_ns={"dram": 123_000.0},
        )
        inp = split_global_memory_paths(
            inp,
            read_l2_bytes=read_bytes,
            read_dram_bytes=read_bytes,
            write_l2_bytes=write_bytes,
            write_dram_bytes=write_bytes,
        )

        with pytest.raises(ValueError, match="reaches multiple memory paths"):
            FORMULA.predict_kernel(inp)

    def test_lower_bound_arbitration_cannot_claim_complete_support(self):
        inp = _make_input(
            demand_per_resource={
                "l2_cache": 1_000_000.0,
                "dram": 1_000_000.0,
            },
            include_dispatch_overhead=False,
            support_level="complete",
            limitations=(),
        )

        with pytest.raises(
                ValueError, match="complete.*lower-bound memory arbitration"):
            FORMULA.predict_kernel(inp)


class TestFormulaNpuPaths:
    def _npu_input(
            self, *, active_units, read_bytes=0.0,
            read_dram_bytes=0.0, write_bytes=0.0,
            write_dram_bytes=None, write_l2_bytes=None):
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=active_units,
            grid_size=active_units,
            demand_per_resource={
                "l2_cache": read_bytes,
                "dram": read_dram_bytes,
                "mte3": write_bytes,
            },
            include_dispatch_overhead=False,
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
        )
        if write_bytes > 0:
            inp = append_npu_write_path(
                inp,
                NPU_HW,
                dram_write_bytes=write_dram_bytes,
                l2_write_bytes=write_l2_bytes,
            )
        return inp

    @pytest.mark.parametrize(
        ("task_kind", "grid_size", "expected_ns"),
        (("aic", 8, 1040),
         ("aiv", 8, 1627.6295411781734),
         ("mixed", 8, 4960)),
    )
    def test_task_duration_floor_overlaps_resource_execution(
            self, task_kind, grid_size, expected_ns):
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=grid_size,
            grid_size=grid_size,
            block_size=1,
            launch_task_kind=task_kind,
            include_dispatch_overhead=True,
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
        )

        prediction = NPU_FORMULA.predict_kernel(inp)

        if task_kind == "aiv":
            expected_submit_ns = (
                NPU_HW.aiv_nonempty_task_latency_model["submit_ns"])
            assert prediction.t_launch_s * 1e9 == pytest.approx(
                expected_submit_ns)
        else:
            assert prediction.t_launch_s == 0.0
        assert prediction.t_total_ns == pytest.approx(expected_ns)
        assert prediction.bottleneck_group == "launch"

    def test_aiv_pipeline_workers_wait_for_ordered_descriptors(self):
        worker_count = 2
        compute_ns = 10_000.0
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=worker_count,
            grid_size=worker_count,
            block_size=1,
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            include_dispatch_overhead=True,
            resource_time_overrides_ns={"compute_vector": compute_ns},
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
        )
        tasks = tuple(
            PipelineTaskDemand(
                name=f"worker{worker}",
                worker=worker,
                stages=(PipelineStageDemand(
                    "compute", "compute:compute_vector",
                    1.0 / worker_count),),
            )
            for worker in range(worker_count)
        )
        inp = attach_pipeline_work_plan(
            inp,
            NPU_HW,
            PipelineWorkPlan(worker_count, tasks),
            promote_npu_partial=True,
        )

        prediction = NPU_FORMULA.predict_kernel(inp)

        scheduler = NPU_HW.aiv_nonempty_task_latency_model
        submit_ns = scheduler["submit_ns"]
        descriptor_ns = scheduler["descriptor_ns_per_aiv"]
        expected_ns = submit_ns + worker_count * descriptor_ns + compute_ns
        assert prediction.diagnostics.pipeline.worker_makespan_s == {
            "0": pytest.approx(compute_ns * 1e-9),
            "1": pytest.approx(compute_ns * 1e-9),
        }
        assert prediction.t_launch_s * 1e9 == pytest.approx(submit_ns)
        assert prediction.t_total_ns == pytest.approx(expected_ns)

    def test_aiv_descriptor_and_tiling_prologue_form_worker_pipeline(self):
        worker_count = 3
        compute_ns = 10_000.0
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=worker_count,
            grid_size=worker_count,
            block_size=1,
            demand_per_resource={"compute_vector": 3.0},
            launch_task_kind="aiv",
            include_dispatch_overhead=True,
            resource_time_overrides_ns={"compute_vector": compute_ns},
            task_prologue_contract=(
                CANN_AIV_TILING_MATERIALIZATION_PROLOGUE),
            task_prologue_repetitions=1,
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
        )
        tasks = tuple(
            PipelineTaskDemand(
                name=f"worker{worker}",
                worker=worker,
                stages=(PipelineStageDemand(
                    "compute", "compute:compute_vector",
                    1.0 / worker_count),),
            )
            for worker in range(worker_count)
        )
        inp = attach_pipeline_work_plan(
            inp,
            NPU_HW,
            PipelineWorkPlan(worker_count, tasks),
            promote_npu_partial=True,
        )
        inp = dataclasses.replace(
            inp, aiv_descriptor_prologue_pipeline=True)

        prediction = NPU_FORMULA.predict_kernel(inp)

        scheduler = NPU_HW.aiv_nonempty_task_latency_model
        prologue_ready_ns = None
        previous_prologue_ns = 0.0
        for worker in range(worker_count):
            cumulative_prologue_ns = NPU_HW.task_prologue_duration_ns(
                CANN_AIV_TILING_MATERIALIZATION_PROLOGUE, worker + 1)
            descriptor_ready_ns = (
                scheduler["submit_ns"]
                + (worker + 1) * scheduler["descriptor_ns_per_aiv"])
            prologue_ready_ns = (
                max(descriptor_ready_ns, prologue_ready_ns)
                if prologue_ready_ns is not None else descriptor_ready_ns)
            prologue_ready_ns += (
                cumulative_prologue_ns - previous_prologue_ns)
            previous_prologue_ns = cumulative_prologue_ns
        expected_ns = prologue_ready_ns + compute_ns
        assert prediction.t_total_ns == pytest.approx(expected_ns)

    def test_aiv_complete_task_floor_only_is_not_decomposed(self):
        worker_count = 2
        compute_ns = 10_000.0
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=worker_count,
            grid_size=worker_count,
            block_size=1,
            launch_grid=(worker_count, 1, 1),
            launch_block=(1, 1, 1),
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            include_dispatch_overhead=True,
            resource_time_overrides_ns={"compute_vector": compute_ns},
            task_duration_floor_contract=AIV_COMPLETE_TASK_FLOOR_ONLY,
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
        )
        tasks = tuple(
            PipelineTaskDemand(
                name=f"worker{worker}",
                worker=worker,
                stages=(PipelineStageDemand(
                    "compute", "compute:compute_vector",
                    1.0 / worker_count),),
            )
            for worker in range(worker_count)
        )
        inp = attach_pipeline_work_plan(
            inp,
            NPU_HW,
            PipelineWorkPlan(worker_count, tasks),
            promote_npu_partial=True,
        )

        prediction = NPU_FORMULA.predict_kernel(inp)

        assert prediction.diagnostics.pipeline.worker_makespan_s == {
            "0": pytest.approx(compute_ns * 1e-9),
            "1": pytest.approx(compute_ns * 1e-9),
        }
        assert prediction.t_launch_s == 0.0
        assert prediction.t_total_ns == pytest.approx(compute_ns)

    def test_aiv_worker_start_fill_precedes_balanced_pipeline_completion(self):
        worker_count = 3
        compute_ns = 10_000.0
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=worker_count,
            grid_size=worker_count,
            block_size=1,
            launch_grid=(worker_count, 1, 1),
            launch_block=(1, 1, 1),
            demand_per_resource={"compute_vector": 3.0},
            launch_task_kind="aiv",
            include_dispatch_overhead=True,
            resource_time_overrides_ns={"compute_vector": compute_ns},
            task_duration_floor_contract=AIV_COMPLETE_TASK_FLOOR_ONLY,
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
        )
        tasks = tuple(
            PipelineTaskDemand(
                name=f"worker{worker}",
                worker=worker,
                stages=(PipelineStageDemand(
                    "compute", "compute:compute_vector",
                    1.0 / worker_count),),
            )
            for worker in range(worker_count)
        )
        inp = attach_pipeline_work_plan(
            inp,
            NPU_HW,
            PipelineWorkPlan(worker_count, tasks),
            promote_npu_partial=True,
        )
        inp = dataclasses.replace(
            inp, aiv_worker_start_contract=AIV_WORKER_START_FILL)

        prediction = NPU_FORMULA.predict_kernel(inp)

        assert prediction.t_launch_s == 0.0
        assert prediction.t_total_ns == pytest.approx(
            compute_ns + (worker_count - 1) * 40.0)

    def test_aiv_source_entry_fill_follows_legacy_descriptor_pipeline(self):
        worker_count = 3
        compute_ns = 10_000.0
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=worker_count,
            grid_size=worker_count,
            block_size=1,
            launch_grid=(worker_count, 1, 1),
            launch_block=(1, 1, 1),
            demand_per_resource={"compute_vector": 3.0},
            launch_task_kind="aiv",
            include_dispatch_overhead=True,
            resource_time_overrides_ns={"compute_vector": compute_ns},
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
        )
        tasks = tuple(
            PipelineTaskDemand(
                name=f"worker{worker}",
                worker=worker,
                stages=(PipelineStageDemand(
                    "compute", "compute:compute_vector",
                    1.0 / worker_count),),
            )
            for worker in range(worker_count)
        )
        inp = attach_pipeline_work_plan(
            inp,
            NPU_HW,
            PipelineWorkPlan(worker_count, tasks),
            promote_npu_partial=True,
        )
        inp = dataclasses.replace(
            inp, aiv_source_entry_fill_contract=AIV_SOURCE_ENTRY_FILL)

        prediction = NPU_FORMULA.predict_kernel(inp)

        scheduler = NPU_HW.aiv_nonempty_task_latency_model
        assert scheduler is not None
        assert prediction.t_total_ns == pytest.approx(
            scheduler["submit_ns"]
            + worker_count * scheduler["descriptor_ns_per_aiv"]
            + compute_ns
            + (worker_count - 1) * 40.0)

    def test_aiv_source_pipeline_uses_only_submission_prefix(self):
        worker_count = 2
        compute_ns = 10_000.0
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=worker_count,
            grid_size=worker_count,
            block_size=1,
            launch_grid=(worker_count, 1, 1),
            launch_block=(1, 1, 1),
            demand_per_resource={"compute_vector": 2.0},
            launch_task_kind="aiv",
            include_dispatch_overhead=True,
            resource_time_overrides_ns={"compute_vector": compute_ns},
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
        )
        tasks = tuple(
            PipelineTaskDemand(
                name=f"worker{worker}",
                worker=worker,
                stages=(PipelineStageDemand(
                    "compute", "compute:compute_vector",
                    1.0 / worker_count),),
            )
            for worker in range(worker_count)
        )
        inp = attach_pipeline_work_plan(
            inp,
            NPU_HW,
            PipelineWorkPlan(worker_count, tasks),
            promote_npu_partial=True,
        )
        inp = dataclasses.replace(
            inp,
            task_duration_floor_contract=AIV_SOURCE_PIPELINE_LAUNCH_ONLY,
        )

        prediction = NPU_FORMULA.predict_kernel(inp)

        submit_ns = NPU_HW.aiv_nonempty_task_latency_model["submit_ns"]
        assert prediction.t_launch_s * 1e9 == pytest.approx(submit_ns)
        assert prediction.t_total_ns == pytest.approx(
            submit_ns + compute_ns)

    def test_aiv_source_pipeline_can_use_explicit_worker_start_fill(self):
        worker_count = 3
        compute_ns = 10_000.0
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=worker_count,
            grid_size=worker_count,
            block_size=1,
            launch_grid=(worker_count, 1, 1),
            launch_block=(1, 1, 1),
            demand_per_resource={"compute_vector": 3.0},
            launch_task_kind="aiv",
            include_dispatch_overhead=True,
            resource_time_overrides_ns={"compute_vector": compute_ns},
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
            task_prologue_contract=(
                CANN_AIV_TILING_MATERIALIZATION_PROLOGUE),
            task_prologue_repetitions=1,
        )
        tasks = tuple(
            PipelineTaskDemand(
                name=f"worker{worker}",
                worker=worker,
                stages=(PipelineStageDemand(
                    "compute", "compute:compute_vector",
                    1.0 / worker_count),),
            )
            for worker in range(worker_count)
        )
        inp = attach_pipeline_work_plan(
            inp,
            NPU_HW,
            PipelineWorkPlan(worker_count, tasks),
            promote_npu_partial=True,
        )
        inp = dataclasses.replace(
            inp,
            task_duration_floor_contract=AIV_SOURCE_PIPELINE_LAUNCH_ONLY,
            aiv_worker_start_contract=AIV_WORKER_START_FILL,
        )

        prediction = NPU_FORMULA.predict_kernel(inp)

        prologue_ns = NPU_HW.task_prologue_duration_ns(
            CANN_AIV_TILING_MATERIALIZATION_PROLOGUE, worker_count)
        worker_fill_ns = NPU_HW.aiv_worker_start_fill_ns(worker_count)
        assert prediction.t_launch_s * 1e9 == pytest.approx(prologue_ns)
        assert prediction.t_total_ns == pytest.approx(
            compute_ns + max(prologue_ns, worker_fill_ns))

    def test_enabled_npu_launch_floor_fails_loud_when_curve_is_absent(
            self, monkeypatch):
        monkeypatch.setattr(NPU_HW, "aiv_empty_kernel_duration_lut", None)
        monkeypatch.setattr(
            NPU_HW, "aiv_nonempty_task_latency_model", None)
        monkeypatch.setattr(
            NPU_HW, "aiv_memory_task_latency_model", None)
        inp = _make_input(
            contract_hardware=NPU_HW,
            active_units=8,
            grid_size=8,
            block_size=1,
            launch_task_kind="aiv",
            include_dispatch_overhead=True,
            limitations=(
                "synthetic_test_contract",
                "workload_producer_independence_unavailable",
            ),
        )

        with pytest.raises(ValueError, match="no AIV task duration floor"):
            NPU_FORMULA.predict_kernel(inp)

    def test_mte2_and_mte3_are_distinct_paths_with_directional_gm_capacity(self):
        active_units = 8
        combined = NPU_FORMULA.predict_kernel(self._npu_input(
            active_units=active_units,
            read_bytes=8_000_000.0,
            read_dram_bytes=8_000_000.0,
            write_bytes=8_000_000.0,
        ))

        paths = {item.path_name for item in combined.diagnostics.memory}
        assert paths == {"mte2_read", "mte3_write"}
        assert all(
            item.rate_unit == "bytes"
            and item.initial_transactions is None
            and item.first_request_latency_s is None
            for item in combined.diagnostics.memory)
        dram = NPU_HW.resource_by_role("dram").name
        assert combined.t_per_resource[dram] > 0.0
        arbitration = combined.diagnostics.memory_arbitration[0]
        common_capacity = NPU_HW.dram_capacity_for_access(
            "read",
            active_units=active_units,
            sm_freq_hz=NPU_HW.nominal_sm_freq_hz,
            dram_freq_hz=NPU_HW.nominal_dram_freq_hz,
        )
        assert arbitration.arbitration_mode == (
            "shared_directional_capacity")
        assert arbitration.analytical_time_s == pytest.approx(
            (arbitration.read_bytes + arbitration.write_bytes)
            / common_capacity)
        assert combined.t_per_resource[dram] == pytest.approx(
            arbitration.analytical_time_s)

    def test_mte3_temporary_bytes_need_not_claim_a_dram_continuation(self):
        inp = self._npu_input(
            active_units=8,
            read_bytes=4_000.0,
            read_dram_bytes=2_000.0,
            write_bytes=3_000.0,
            write_dram_bytes=1_000.0,
        )
        write_path, = tuple(
            path for path in inp.memory_paths if path.name == "mte3_write")
        nodes = {
            node.resource_role: node.bytes for node in write_path.nodes}

        assert nodes == {"mte3": 3_000.0, "dram": 1_000.0}
        assert inp.demand_per_resource["mte3"] == 3_000.0
        assert inp.demand_per_resource["dram"] == 3_000.0
        NPU_FORMULA.predict_kernel(inp)

    def test_mte3_proven_l2_continuation_is_a_shared_write_service(self):
        inp = self._npu_input(
            active_units=8,
            write_bytes=3_000.0,
            write_l2_bytes=2_000.0,
            write_dram_bytes=1_000.0,
        )
        write_path, = tuple(
            path for path in inp.memory_paths if path.name == "mte3_write")
        nodes = {
            node.resource_role: node.bytes for node in write_path.nodes}

        assert nodes == {
            "mte3": 3_000.0,
            "l2_cache": 2_000.0,
            "dram": 1_000.0,
        }
        assert inp.demand_per_resource["l2_cache"] == 2_000.0
        assert inp.demand_per_resource["dram"] == 1_000.0
        NPU_FORMULA.predict_kernel(inp)

    @pytest.mark.parametrize(
        ("write_l2_bytes", "write_dram_bytes", "message"),
        (
            (3_001.0, 1_000.0, "L2 continuation"),
            (2_000.0, 2_001.0, "DRAM continuation"),
        ),
    )
    def test_mte3_write_continuations_fail_loudly(
            self, write_l2_bytes, write_dram_bytes, message):
        with pytest.raises(ValueError, match=message):
            self._npu_input(
                active_units=8,
                write_bytes=3_000.0,
                write_l2_bytes=write_l2_bytes,
                write_dram_bytes=write_dram_bytes,
            )

    def test_mte3_capacity_scales_with_active_aiv_cores(self):
        write_bytes = 48_000_000.0
        one = NPU_FORMULA.predict_kernel(self._npu_input(
            active_units=1, write_bytes=write_bytes))
        two = NPU_FORMULA.predict_kernel(self._npu_input(
            active_units=2, write_bytes=write_bytes))

        assert one.t_total_s == pytest.approx(2.0 * two.t_total_s)

    def test_unstructured_mte3_requires_explicit_oracle_pipe_override(self):
        base = self._npu_input(active_units=8, read_bytes=8_000_000.0)
        flat = dict(base.demand_per_resource)
        flat["mte3"] = 8192.0
        unsupported = dataclasses.replace(base, demand_per_resource=flat)

        with pytest.raises(ValueError, match="explicit Oracle"):
            NPU_FORMULA.predict_kernel(unsupported)

        oracle = dataclasses.replace(
            unsupported,
            resource_time_overrides_ns={"mte3": 100.0},
        )
        prediction = NPU_FORMULA.predict_kernel(oracle)

        assert prediction.t_per_resource[
            NPU_HW.resource_by_role("mte3").name] == pytest.approx(100e-9)


class TestFormulaOverrides:
    def test_observed_compute_time_replaces_analytical_time(self):
        inp = _make_input(
            demand_per_resource={"compute_tensor": 1.0},
            resource_time_overrides_ns={"compute_tensor": 50_000.0},
            include_dispatch_overhead=False,
        )
        prediction = FORMULA.predict_kernel(inp)
        tensor = HW.resource_by_role("compute_tensor").name

        assert prediction.t_per_resource[tensor] == pytest.approx(50e-6)
        assert prediction.t_total_ns == pytest.approx(50_000.0)

    def test_unknown_override_role_fails_loudly(self):
        inp = _make_input(resource_time_overrides_ns={"unknown_pipe": 1.0})
        with pytest.raises(
                ValueError, match="time-override roles.*unknown_pipe"):
            FORMULA.predict_kernel(inp)


class TestPredictionInputContract:
    def test_core_fields_have_no_defaults(self):
        parameters = inspect.signature(PredictionInput).parameters
        for field in (
                "eta", "wave_efficiency", "variant", "mem_model",
                "compute_demands", "memory_paths", "pipeline_plan"):
            assert parameters[field].default is inspect.Parameter.empty

    def test_task_duration_floor_contract_rejects_unknown_identity(self):
        with pytest.raises(ValueError, match="is unsupported"):
            _make_input(task_duration_floor_contract="kernel_name_floor")

    def test_task_duration_floor_contract_requires_full_gpu_launch(self):
        contract = "warm_default_cache_memory_dependency_task_v1"
        with pytest.raises(ValueError, match="full launch geometry"):
            _make_input(task_duration_floor_contract=contract)
        with pytest.raises(ValueError, match="generic gpu_mshr launch"):
            _make_input(
                contract_hardware=NPU_HW,
                launch_grid=(48, 1, 1),
                launch_block=(1, 1, 1),
                task_duration_floor_contract=contract,
            )
        with pytest.raises(ValueError, match="launch terms to be enabled"):
            _make_input(
                launch_grid=(108, 1, 1),
                launch_block=(32, 1, 1),
                task_duration_floor_contract=contract,
                include_dispatch_overhead=False,
            )

    def test_npu_partial_support_requires_worker_pipeline(self):
        with pytest.raises(
                ValueError,
                match="partial prediction requires a source-backed worker"):
            _make_input(
                contract_hardware=NPU_HW,
                support_level="partial",
            )

    def test_generic_helper_cannot_promote_npu_pipeline_support(self):
        inp = _make_input(
            contract_hardware=NPU_HW,
            demand_per_resource={"compute_vector": 1.0},
            include_dispatch_overhead=False,
        )

        with pytest.raises(
                ValueError,
                match="requires explicit partial-support promotion"):
            attach_uniform_pipeline(
                inp,
                NPU_HW,
                task_count=1,
                stages=((
                    "compute", "compute:compute_vector", False),),
                buffer_slots=1,
            )

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_demand_fails_loudly(self, value):
        with pytest.raises(ValueError, match="finite non-negative"):
            _make_input(demand_per_resource={"dram": value})

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.0])
    def test_invalid_execution_time_floor_fails_loudly(self, value):
        with pytest.raises(ValueError, match="execution_time_floor_ns"):
            _make_input(execution_time_floor_ns=value)

    def test_execution_time_floor_competes_with_resource_bounds(self):
        inp = _make_input(
            include_dispatch_overhead=False,
            execution_time_floor_ns=1234.0,
        )

        prediction = PerformanceFormula(HW).predict_kernel(inp)

        assert prediction.t_total_ns == pytest.approx(1234.0)
        assert prediction.bottleneck_group == "analytical_execution_floor"

    def test_unknown_resource_role_fails_loudly(self):
        demand = _empty_demand()
        del demand["warp_issue"]
        demand["typo_resource"] = 0.0
        with pytest.raises(ValueError, match="schema mismatch"):
            PredictionInput(
                demand_per_resource=demand,
                eta=1.0,
                wave_efficiency=1.0,
                active_units=HW.num_compute_units,
                sm_freq_hz=HW.nominal_sm_freq_hz,
                dram_freq_hz=HW.nominal_dram_freq_hz,
                grid_size=HW.num_compute_units,
                block_size=HW.parallel_width,
                variant="test",
                mem_model="gpu_mshr",
                compute_demands=(),
                memory_paths=(),
                pipeline_plan=None,
            )

    @pytest.mark.parametrize("field", ["eta", "wave_efficiency"])
    def test_numeric_fields_reject_strings(self, field):
        with pytest.raises(ValueError, match="numeric fields must be finite"):
            _make_input(**{field: "1.0"})

    @pytest.mark.parametrize("field", ["active_units", "grid_size", "block_size"])
    @pytest.mark.parametrize("value", [True, 1.0, "1"])
    def test_integer_fields_reject_coercible_types(self, field, value):
        with pytest.raises(TypeError, match=f"{field} must be an integer"):
            _make_input(**{field: value})
