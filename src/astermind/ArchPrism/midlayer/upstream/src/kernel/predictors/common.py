"""Shared physical model input and arithmetic for pure predictors."""

import dataclasses
import math
import threading
from fractions import Fraction
from typing import Dict, Mapping, Optional, Tuple, Union

from src.kernel.kernel_spec import (
    KernelPrediction, KernelSpec, graph_replayed_launch, prediction_from_ns,
)
from src.kernel.model_contract import (
    PredictionInput,
    AIV_COMPLETE_TASK_FLOOR_ONLY,
    AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    AIV_SOURCE_PIPELINE_LAUNCH_ONLY,
    AIV_SOURCE_ENTRY_FILL,
    AIV_WORKER_START_FILL,
    GPU_SOURCE_PIPELINE_OUTER_ENVELOPE,
    AnalyticalDependencyPath,
    AnalyticalPhaseLedger,
    AttentionPipelineEnvelopeDemand,
    BarrierPipelineEnvelopeDemand,
    CannGroupedMatmulEnvelopeDemand,
    CannSingleCoreSplitKEnvelopeDemand,
    CANN_AIV_TILING_MATERIALIZATION_48B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_56B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_80B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_192B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_PROLOGUE,
    ComputeDemand,
    ComputeServiceDemand,
    FiaCrossDomainWorkPlan,
    FiaSplitFuseWorkPlan,
    GpuGroupedGemmPipelineEnvelopeDemand,
    MemoryAccessClassDemand,
    MemoryNodeDemand,
    MemoryPathDemand,
    MemorySourceDemand,
    MetaxGroupedOperandPipelineDemand,
    PipelineBuffer,
    PipelineStageDemand,
    PipelineTaskDemand,
    PipelineTaskProgramDemand,
    PipelineTaskSequenceDemand,
    PipelineWorkPlan,
    TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR,
    TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR,
    PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
    PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS,
    WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
)


# The production model intentionally makes no per-kernel residency estimate.
# Grid tail is modeled separately; resource occupancy is explicitly full until
# a source-bound physical residency contract is introduced.
FULL_RESOURCE_OCCUPANCY = 1.0


_PIPELINE_STAGE_TEMPLATE_CACHE_SIZE = 4096
_pipeline_stage_templates = {}
_pipeline_stage_templates_lock = threading.Lock()
_TYPED_COMPUTE_PIPELINE_STAGE_TEMPLATE_CACHE_SIZE = 4096
_typed_compute_pipeline_stage_templates = {}
_typed_compute_pipeline_stage_templates_lock = threading.Lock()
_MEMORY_ACCESS_TEMPLATE_CACHE_SIZE = 4096
_memory_access_templates = {}
_memory_access_templates_lock = threading.Lock()


def bind_pipeline_stage_template(
        name, service, work_fraction, include_first_request_latency=False,
        *, predecessors=None, start_predecessors=(), prior_resources=(),
        absolute_duration_s=None):
    """Bind dynamic source work to a shared, validated stage template.

    A hardware adapter owns the launch/source facts which choose the stage
    name, service and dependency topology.  A800, C500 and 910B adapters can
    therefore use the same cache and fail-closed binding path while retaining
    their own physical launch rules.  The cache contains no shape, byte,
    duration or prediction value: each invocation validates and binds exactly
    one dynamic work fraction or mechanism duration.
    """

    is_absolute = absolute_duration_s is not None
    if is_absolute == (work_fraction is not None):
        raise ValueError(
            "pipeline stage template requires exactly one dynamic work or "
            "duration")
    key = (
        name, service, include_first_request_latency, predecessors,
        start_predecessors, prior_resources, is_absolute,
    )
    with _pipeline_stage_templates_lock:
        if key not in _pipeline_stage_templates:
            if (len(_pipeline_stage_templates)
                    >= _PIPELINE_STAGE_TEMPLATE_CACHE_SIZE):
                _pipeline_stage_templates.clear()
            _pipeline_stage_templates[key] = PipelineStageDemand(
                name, service, None if is_absolute else 1.0,
                include_first_request_latency,
                predecessors=predecessors,
                start_predecessors=start_predecessors,
                prior_resources=prior_resources,
                absolute_duration_s=1.0 if is_absolute else None,
            )
        template = _pipeline_stage_templates[key]
    return PipelineStageDemand.bind_validated_dynamic_value(
        template,
        work_fraction=work_fraction,
        absolute_duration_s=absolute_duration_s,
    )


def bind_typed_compute_pipeline_stage_template(
        name, service, typed_services, operations, *,
        serial_invocations=1, predecessors=None, start_predecessors=(),
        prior_resources=()):
    """Bind dynamic typed compute work to a reusable source-stage contract.

    Like ``bind_pipeline_stage_template``, the cache stores static source
    topology only.  ``operations`` are validated and rebound for each
    prediction, never used as duration or prediction-result keys.  It is
    shared by all hardware adapters whose source programs use typed compute
    services, while launch/residency facts remain adapter-owned.
    """

    if (not isinstance(typed_services, tuple) or not typed_services
            or any(not isinstance(value, str) or not value
                   for value in typed_services)):
        raise TypeError("typed compute stage services must be non-empty strings")
    if len(typed_services) != len(set(typed_services)):
        raise ValueError("typed compute stage services cannot repeat")
    if (isinstance(serial_invocations, bool)
            or not isinstance(serial_invocations, int)
            or serial_invocations <= 0):
        raise ValueError(
            "typed compute stage serial invocations must be positive integer")
    if (not isinstance(operations, tuple)
            or len(operations) != len(typed_services)):
        raise ValueError(
            "typed compute stage operations must align with services")
    key = (
        name, service, typed_services, serial_invocations,
        predecessors, start_predecessors, prior_resources,
    )
    with _typed_compute_pipeline_stage_templates_lock:
        if key not in _typed_compute_pipeline_stage_templates:
            if (len(_typed_compute_pipeline_stage_templates)
                    >= _TYPED_COMPUTE_PIPELINE_STAGE_TEMPLATE_CACHE_SIZE):
                _typed_compute_pipeline_stage_templates.clear()
            _typed_compute_pipeline_stage_templates[key] = PipelineStageDemand(
                name, service, None,
                compute_service_demands=tuple(
                    ComputeServiceDemand(
                        typed_service, 1.0,
                        serial_invocations=serial_invocations)
                    for typed_service in typed_services),
                predecessors=predecessors,
                start_predecessors=start_predecessors,
                prior_resources=prior_resources,
            )
        template = _typed_compute_pipeline_stage_templates[key]
    return PipelineStageDemand.bind_validated_typed_compute_operations(
        template, operations)


def bind_memory_access_class_template(
        service, ingress_resource_role, logical_bytes, signature,
        replaces_ingress_path_service=False):
    """Bind dynamic bytes to a shared checked access-class source template.

    A hardware adapter selects the source service and immutable signature;
    only the current shape's logical bytes are dynamic.  This is the typed
    memory counterpart to ``bind_pipeline_stage_template`` and contains no
    traffic prediction, duration, or workload-result cache.
    """

    key = (
        service,
        ingress_resource_role,
        signature,
        replaces_ingress_path_service,
    )
    with _memory_access_templates_lock:
        if key not in _memory_access_templates:
            if (len(_memory_access_templates)
                    >= _MEMORY_ACCESS_TEMPLATE_CACHE_SIZE):
                _memory_access_templates.clear()
            _memory_access_templates[key] = MemoryAccessClassDemand(
                service=service,
                ingress_resource_role=ingress_resource_role,
                logical_bytes=1.0,
                signature=signature,
                replaces_ingress_path_service=replaces_ingress_path_service,
            )
        template = _memory_access_templates[key]
    return MemoryAccessClassDemand.bind_validated_logical_bytes(
        template,
        logical_bytes=logical_bytes,
    )


def attach_structured_contract(
        inp: PredictionInput, hardware, *,
        transaction_bytes: Optional[float],
        path_name: Optional[str],
        access_kind: str = "read",
        source_producers_per_unit: float = 1.0,
        independent_requests_per_producer: Optional[float] = 1.0,
        frontend_rate_per_unit_s: Optional[float] = None,
        prepend_npu_aiv_mte2_ingress: bool = False) -> PredictionInput:
    """Attach the one canonical compute and memory-path contract.

    Per-kind predictors still own physical operation/traffic derivation.  This
    function only normalizes those already-derived quantities into the shared
    solver schema; it does not infer a kernel variant or fit a duration.
    """

    import dataclasses

    if not isinstance(inp, PredictionInput):
        raise TypeError("attach_structured_contract requires PredictionInput")
    if inp.compute_demands or inp.memory_paths or inp.pipeline_plan is not None:
        raise ValueError(
            "structured contract must be attached exactly once")
    if not isinstance(prepend_npu_aiv_mte2_ingress, bool):
        raise TypeError("prepend_npu_aiv_mte2_ingress must be boolean")
    demand = dict(inp.demand_per_resource)
    if not hardware.l1_cache_enabled and inp.demand_per_resource["l1_cache"]:
        demand["l1_cache"] = 0.0
    if inp.mem_model == "npu_tile":
        # The aggregate-only NPU contract currently establishes Cube/AIV
        # operations plus L2/GM/MTE3 byte demand.  GPU-derived L1, SMEM and
        # crossbar traffic have no source-backed AIC/UB tile lifecycle and
        # must not be priced through similarly named HWSpec resources.
        for role in ("l1_cache", "smem", "xbar"):
            demand[role] = 0.0
    compute_demands = []
    for role in (
            "compute_tensor", "compute_vector", "compute_reduction",
            "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
            "scalar_aiv"):
        if demand[role] <= 0:
            continue
        compute_demands.append(ComputeDemand(
            resource_role=role,
            operations=float(demand[role]),
            shape_efficiency=1.0,
            schedule_efficiency=(
                inp.eta * inp.wave_efficiency),
        ))
    compute_demands = tuple(compute_demands)

    l2_bytes = float(demand["l2_cache"])
    dram_bytes = float(demand["dram"])
    if l2_bytes == 0.0:
        if dram_bytes != 0.0:
            raise ValueError(
                "positive DRAM traffic requires a structured ingress path")
        memory_paths = ()
    elif inp.mem_model in ("gpu_mshr", "npu_tile"):
        if inp.mem_model == "gpu_mshr":
            if transaction_bytes is not None:
                raise ValueError(
                    "GPU transaction sizes come from explicit per-level "
                    "HWSpec fields, not cacheline_bytes")
            transaction_bytes_by_role = {
                "l2_cache": float(hardware.l2_transaction_bytes),
                "dram": float(hardware.dram_transaction_bytes),
            }
        elif inp.mem_model == "npu_tile":
            if transaction_bytes is not None:
                raise ValueError(
                    "NPU transaction size must remain None until physical "
                    "MTE transaction geometry is independently established")
            transaction_bytes_by_role = {
                "l2_cache": None,
                "dram": None,
            }
        else:
            raise AssertionError(
                f"unhandled target memory model: {inp.mem_model!r}")
        if path_name is None or not isinstance(path_name, str) or not path_name:
            raise ValueError(
                "target memory path requires a non-empty path name")
        l2_transaction_bytes = transaction_bytes_by_role["l2_cache"]
        nodes = [MemoryNodeDemand(
            resource_role="l2_cache",
            transactions=(
                None if l2_transaction_bytes is None
                else l2_bytes / l2_transaction_bytes),
            bytes=l2_bytes,
            transaction_bytes=l2_transaction_bytes,
        )]
        if prepend_npu_aiv_mte2_ingress:
            if inp.mem_model != "npu_tile":
                raise ValueError(
                    "NPU AIV MTE2 ingress requires npu_tile memory")
            if hardware.kernel_memory_model != "npu_tile":
                raise ValueError(
                    "NPU AIV MTE2 ingress requires npu_tile hardware")
            if inp.launch_task_kind != "aiv":
                raise ValueError(
                    "NPU AIV MTE2 ingress requires an AIV-only task")
            mte2_units = hardware.resource_by_role("mte2").total_units
            if inp.active_units > mte2_units:
                raise ValueError(
                    "NPU MTE2 active units exceed the physical AIV domain: "
                    f"active={inp.active_units}, physical={mte2_units}")
            nodes.insert(0, MemoryNodeDemand(
                resource_role="mte2",
                transactions=None,
                bytes=l2_bytes,
                transaction_bytes=None,
            ))
        if dram_bytes > 0:
            dram_transaction_bytes = transaction_bytes_by_role["dram"]
            nodes.append(MemoryNodeDemand(
                resource_role="dram",
                transactions=(
                    None if dram_transaction_bytes is None
                    else dram_bytes / dram_transaction_bytes),
                bytes=dram_bytes,
                transaction_bytes=dram_transaction_bytes,
            ))
        memory_paths = (MemoryPathDemand(
            name=path_name,
            access_kind=access_kind,
            initial_transactions=(
                None if l2_transaction_bytes is None
                else l2_bytes / l2_transaction_bytes),
            nodes=tuple(nodes),
            source=MemorySourceDemand(
                active_units=inp.active_units,
                producers_per_unit=source_producers_per_unit,
                independent_requests_per_producer=(
                    independent_requests_per_producer),
                frontend_rate_per_unit_s=frontend_rate_per_unit_s,
            ),
        ),)
    elif inp.mem_model in ("pim_bank", "systolic_scratchpad"):
        # These explicit compatibility architectures are outside the current
        # three-hardware acceptance scope and retain their existing physical
        # formula until a transaction contract exists for their memory path.
        if transaction_bytes is not None or path_name is not None:
            raise ValueError(
                "compatibility memory models cannot declare target paths")
        memory_paths = ()
    else:
        raise ValueError(f"unknown PredictionInput.mem_model: {inp.mem_model!r}")
    return dataclasses.replace(
        inp,
        demand_per_resource=demand,
        compute_demands=compute_demands,
        memory_paths=memory_paths,
    )


def replace_structured_demands(
        inp: PredictionInput, replacements: Mapping[str, float]
        ) -> PredictionInput:
    """Replace resource demand while preserving the structured contract.

    This is the canonical observation/ablation boundary.  A replacement of a
    flat compatibility demand must update the corresponding typed compute
    demand and every structured memory node; otherwise the formula would price
    one workload while diagnostics described another one.
    """

    import dataclasses

    if not isinstance(inp, PredictionInput):
        raise TypeError("replace_structured_demands requires PredictionInput")
    if not isinstance(replacements, Mapping):
        raise TypeError("structured demand replacements must be a mapping")
    unknown = set(replacements) - set(inp.demand_per_resource)
    if unknown:
        raise ValueError(
            "structured demand replacement has unknown roles: "
            f"{sorted(unknown)}")
    normalized = {}
    for role, value in replacements.items():
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0):
            raise ValueError(
                "structured demand replacements must be finite and "
                f"non-negative: {role}={value!r}")
        normalized[role] = float(value)

    annotated_ingress_roles = {
        path.nodes[0].resource_role
        for path in inp.memory_paths
        if path.access_class_demands
    }
    replaced_annotated_ingress = (
        annotated_ingress_roles.intersection(normalized))
    if replaced_annotated_ingress:
        raise ValueError(
            "structured demand replacement cannot replace an ingress with "
            "typed memory access-class demand: "
            f"{sorted(replaced_annotated_ingress)}")

    flat = dict(inp.demand_per_resource)
    flat.update(normalized)
    compute_by_role = {
        demand.resource_role: demand for demand in inp.compute_demands
    }
    typed_pipeline_scale = {}
    compute_roles = {
        "compute_tensor", "compute_vector", "compute_reduction",
        "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
    }

    def _rescale_service(service, scale):
        worker_operations = (
            tuple(
                float(Fraction(value) * scale)
                for value in service.operations_by_worker)
            if service.operations_by_worker else ())
        return dataclasses.replace(
            service,
            operations=float(Fraction(service.operations) * scale),
            operations_by_worker=worker_operations,
        )

    for role in compute_roles.intersection(normalized):
        operations = normalized[role]
        if role in compute_by_role:
            if operations == 0.0:
                del compute_by_role[role]
            else:
                prior = compute_by_role[role]
                if prior.exact_request_demands:
                    exact_operations = math.fsum(
                        demand.operations
                        for demand in prior.exact_request_demands)
                    regular_prior = prior.operations - exact_operations
                    regular_replacement = operations - exact_operations
                    if regular_replacement < 0.0:
                        raise ValueError(
                            "structured demand replacement cannot reduce "
                            "operations below a source-defined exact request "
                            "schedule")
                    if not prior.service_demands:
                        if regular_replacement != 0.0:
                            raise ValueError(
                                "structured demand replacement cannot "
                                "rescale a source-defined exact request "
                                "schedule without a replaceable regular "
                                "service")
                        service_demands = ()
                    else:
                        if regular_prior <= 0.0:
                            raise ValueError(
                                "exact request demand has no positive regular "
                                "service remainder")
                        service_demands = tuple(
                            _rescale_service(
                                service,
                                Fraction(regular_replacement)
                                / Fraction(regular_prior))
                            for service in prior.service_demands
                            if regular_replacement > 0.0)
                    # Exact request sizes and worker ownership are deliberately
                    # preserved.  Oracle/ablation replacements may only alter
                    # the remaining regular service demand.
                else:
                    typed_pipeline_scale[f"compute:{role}"] = (
                        Fraction(operations) / Fraction(prior.operations))
                    service_demands = tuple(
                        _rescale_service(
                            service,
                            Fraction(operations)
                            / Fraction(prior.operations))
                        for service in prior.service_demands)
                compute_by_role[role] = dataclasses.replace(
                    prior,
                    operations=operations,
                    service_demands=service_demands,
                )
        elif operations > 0.0:
            raise ValueError(
                "cannot introduce a new compute role without its shape and "
                f"schedule efficiency contract: {role!r}")

    def _rescale_pipeline_stages(source_stages):
        stages = []
        for stage in source_stages:
            if (not stage.compute_service_demands
                    or stage.service not in typed_pipeline_scale):
                stages.append(stage)
                continue
            scale = typed_pipeline_scale[stage.service]
            stages.append(dataclasses.replace(
                stage,
                compute_service_demands=tuple(
                    _rescale_service(service, scale)
                    for service in stage.compute_service_demands),
            ))
        return tuple(stages)

    def _rescale_pipeline_task(task):
        return dataclasses.replace(
            task, stages=_rescale_pipeline_stages(task.stages))

    def _rescale_pipeline_sequence(sequence):
        return dataclasses.replace(
            sequence,
            tasks=tuple(
                _rescale_pipeline_task(task)
                for task in sequence.tasks),
        )

    if inp.pipeline_plan is None or not typed_pipeline_scale:
        pipeline_plan = inp.pipeline_plan
    elif isinstance(inp.pipeline_plan, AttentionPipelineEnvelopeDemand):
        rescaled_phase_by_demand = {}
        rescaled_phases_by_identity = {}

        def _rescale_attention_phase(phase):
            if phase not in rescaled_phase_by_demand:
                rescaled_phase_by_demand[phase] = dataclasses.replace(
                    phase,
                    stages=_rescale_pipeline_stages(phase.stages),
                )
            return rescaled_phase_by_demand[phase]

        def _rescale_attention_phases(phases):
            identity = id(phases)
            if identity not in rescaled_phases_by_identity:
                rescaled_phases_by_identity[identity] = tuple(
                    _rescale_attention_phase(phase) for phase in phases)
            return rescaled_phases_by_identity[identity]

        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan,
            classes=tuple(
                dataclasses.replace(
                    work_class,
                    phases=_rescale_attention_phases(work_class.phases),
                )
                for work_class in inp.pipeline_plan.classes),
        )
    elif isinstance(inp.pipeline_plan, BarrierPipelineEnvelopeDemand):
        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan,
            classes=tuple(
                _rescale_pipeline_sequence(sequence)
                for sequence in inp.pipeline_plan.classes),
        )
    elif isinstance(inp.pipeline_plan, FiaCrossDomainWorkPlan):
        def _rescale_fia_stage(stage):
            service_name = f"compute:{stage.resource_role}"
            if service_name not in typed_pipeline_scale:
                return stage
            scale = typed_pipeline_scale[service_name]
            return dataclasses.replace(
                stage,
                logical_operations=float(
                    Fraction(stage.logical_operations) * scale),
                physical_operations=float(
                    Fraction(stage.physical_operations) * scale),
                typed_services=tuple(
                    _rescale_service(service, scale)
                    for service in stage.typed_services),
            )

        def _rescale_gqa_source_order(source_order):
            """Keep literal GQA panels consistent with replaced Cube work."""

            if source_order is None:
                return None
            return dataclasses.replace(
                source_order,
                rounds=tuple(
                    dataclasses.replace(
                        round_demand,
                        panels=tuple(
                            dataclasses.replace(
                                panel,
                                mm1=_rescale_fia_stage(panel.mm1),
                                mm2=_rescale_fia_stage(panel.mm2),
                            )
                            for panel in round_demand.panels),
                    )
                    for round_demand in source_order.rounds),
            )

        fia_groups = []
        for group in inp.pipeline_plan.groups:
            tasks = []
            for task in group.tasks:
                tasks.append(dataclasses.replace(
                    task,
                    mm1=_rescale_fia_stage(task.mm1),
                    vec1=_rescale_fia_stage(task.vec1),
                    mm2=_rescale_fia_stage(task.mm2),
                    vec2=_rescale_fia_stage(task.vec2),
                    vec1_update=(
                        None if task.vec1_update is None
                        else _rescale_fia_stage(task.vec1_update)),
                    vec1_peer=(
                        None if task.vec1_peer is None
                        else _rescale_fia_stage(task.vec1_peer)),
                    vec1_peer_update=(
                        None if task.vec1_peer_update is None
                        else _rescale_fia_stage(task.vec1_peer_update)),
                    vec2_peer=(
                        None if task.vec2_peer is None
                        else _rescale_fia_stage(task.vec2_peer)),
                    gqa_source_order=_rescale_gqa_source_order(
                        task.gqa_source_order),
                ))
            fia_groups.append(dataclasses.replace(
                group, tasks=tuple(tasks)))
        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan,
            groups=tuple(fia_groups),
            terminal_vector_stages=tuple(
                _rescale_fia_stage(stage)
                for stage in inp.pipeline_plan.terminal_vector_stages),
            aiv_initialization_stages=tuple(
                _rescale_fia_stage(stage)
                for stage in inp.pipeline_plan.aiv_initialization_stages),
        )
    elif isinstance(inp.pipeline_plan, FiaSplitFuseWorkPlan):
        tensor_scale = (
            typed_pipeline_scale["compute:compute_tensor"]
            if "compute:compute_tensor" in typed_pipeline_scale
            else Fraction(1, 1))
        vector_scale = (
            typed_pipeline_scale["compute:compute_vector"]
            if "compute:compute_vector" in typed_pipeline_scale
            else Fraction(1, 1))
        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan,
            tensor_operation_scale=float(
                Fraction(inp.pipeline_plan.tensor_operation_scale)
                * tensor_scale),
            vector_operation_scale=float(
                Fraction(inp.pipeline_plan.vector_operation_scale)
                * vector_scale),
        )
    elif isinstance(inp.pipeline_plan, CannSingleCoreSplitKEnvelopeDemand):
        # The split-K envelope stores source geometry rather than structured
        # task demands.  Its role durations are supplied separately by the
        # formula, so observation replacements must preserve the plan itself.
        pipeline_plan = inp.pipeline_plan
    else:
        pipeline_entries = []
        for entry in inp.pipeline_plan.tasks:
            if isinstance(entry, PipelineTaskProgramDemand):
                pipeline_entries.append(dataclasses.replace(
                    entry,
                    patterns=tuple(
                        _rescale_pipeline_sequence(sequence)
                        for sequence in entry.patterns),
                ))
            elif isinstance(entry, PipelineTaskSequenceDemand):
                pipeline_entries.append(_rescale_pipeline_sequence(entry))
            elif isinstance(entry, PipelineTaskDemand):
                pipeline_entries.append(_rescale_pipeline_task(entry))
            else:
                raise TypeError(
                    "pipeline plan contains an unsupported task demand: "
                    f"{type(entry).__name__}")
        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan, tasks=tuple(pipeline_entries))

    memory_roles = {
        node.resource_role
        for path in inp.memory_paths
        for node in path.nodes
    }
    replaced_memory_roles = memory_roles.intersection(normalized)
    bytes_by_role = {
        role: sum(
            node.bytes
            for path in inp.memory_paths
            for node in path.nodes
            if node.resource_role == role)
        for role in replaced_memory_roles
    }
    invalid_zero_bases = {
        role for role in replaced_memory_roles
        if bytes_by_role[role] == 0.0 and normalized[role] > 0.0
    }
    if invalid_zero_bases:
        raise ValueError(
            "cannot distribute positive observed demand across an empty "
            f"structured role: {sorted(invalid_zero_bases)}")

    paths = []
    for path in inp.memory_paths:
        nodes = []
        for node in path.nodes:
            role = node.resource_role
            if role not in replaced_memory_roles:
                nodes.append(node)
                continue
            role_total = bytes_by_role[role]
            replacement_total = normalized[role]
            byte_count = (
                0.0 if role_total == 0.0 else
                replacement_total * node.bytes / role_total)
            if byte_count == 0.0:
                continue
            nodes.append(dataclasses.replace(
                node,
                transactions=(
                    None if node.transaction_bytes is None
                    else byte_count / node.transaction_bytes),
                bytes=byte_count,
            ))
        if not nodes:
            continue
        if nodes[0].resource_role != path.nodes[0].resource_role:
            raise ValueError(
                "a structured demand replacement cannot remove a path "
                f"ingress while retaining downstream nodes: {path.name!r}")
        paths.append(dataclasses.replace(
            path,
            initial_transactions=nodes[0].transactions,
            nodes=tuple(nodes),
        ))

    if pipeline_plan is None or isinstance(
            pipeline_plan,
            (
                CannSingleCoreSplitKEnvelopeDemand,
                FiaCrossDomainWorkPlan,
                FiaSplitFuseWorkPlan,
            )):
        pipeline_stages = ()
    elif isinstance(pipeline_plan, AttentionPipelineEnvelopeDemand):
        unique_phases = {}
        for work_class in pipeline_plan.classes:
            for phase in work_class.phases:
                unique_phases[id(phase)] = phase
        pipeline_stages = tuple(
            stage
            for phase in unique_phases.values()
            for stage in phase.stages)
    elif isinstance(pipeline_plan, BarrierPipelineEnvelopeDemand):
        pipeline_stages = tuple(
            stage
            for sequence in pipeline_plan.classes
            for task in sequence.tasks
            for stage in task.stages)
    elif isinstance(pipeline_plan, PipelineWorkPlan):
        pipeline_entries = pipeline_plan.tasks
        logical_pipeline_tasks = tuple(
            logical
            for entry in pipeline_entries
            for sequence in (
                entry.patterns
                if isinstance(entry, PipelineTaskProgramDemand)
                else (entry,))
            for logical in (
                sequence.tasks
                if isinstance(sequence, PipelineTaskSequenceDemand)
                else (sequence,)))
        pipeline_stages = tuple(
            stage
            for task in logical_pipeline_tasks
            for stage in task.stages)
    else:
        raise TypeError(
            "structured demand replacement produced an unknown pipeline plan")
    used_services = {
        stage.service
        for stage in pipeline_stages
    }
    available_services = {
        *(f"compute:{role}" for role in compute_by_role),
        *(f"memory:{path.name}" for path in paths),
    }
    removed_services = {
        service for service in used_services
        if (service.startswith("compute:") or service.startswith("memory:"))
        and service not in available_services
    }
    if removed_services:
        raise ValueError(
            "structured demand replacement removes a live pipeline service: "
            f"{sorted(removed_services)}")

    if isinstance(pipeline_plan, FiaSplitFuseWorkPlan):
        l2_scale = (
            Fraction(normalized["l2_cache"])
            / Fraction(inp.demand_per_resource["l2_cache"])
            if "l2_cache" in normalized else Fraction(1, 1))
        mte3_scale = (
            Fraction(normalized["mte3"])
            / Fraction(inp.demand_per_resource["mte3"])
            if "mte3" in normalized else Fraction(1, 1))
        pipeline_plan = dataclasses.replace(
            pipeline_plan,
            l2_byte_scale=float(
                Fraction(pipeline_plan.l2_byte_scale) * l2_scale),
            mte3_byte_scale=float(
                Fraction(pipeline_plan.mte3_byte_scale) * mte3_scale),
        )

    phase_ledger = (
        None if inp.phase_ledger is None
        else inp.phase_ledger.rescaled(
            tuple(compute_by_role.values()), tuple(paths)))
    return dataclasses.replace(
        inp,
        demand_per_resource=flat,
        compute_demands=tuple(compute_by_role.values()),
        memory_paths=tuple(paths),
        pipeline_plan=pipeline_plan,
        phase_ledger=phase_ledger,
    )


def gpu_memory_source(hardware) -> Tuple[float, Optional[float]]:
    """Return the occupancy-free GPU memory-source contract.

    The model has no source-bound per-kernel resident-warp or independent
    request contract.  Represent one generic producer per active unit and let
    the hardware byte/request capacities price the resource demand.
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise ValueError("gpu_memory_source requires gpu_mshr hardware")
    return 1.0, None


def gpu_partial_limitations(hardware, *additional: str) -> Tuple[str, ...]:
    """Describe evidence gaps from capabilities, never hardware identity."""

    if any(not isinstance(value, str) or not value for value in additional):
        raise ValueError("additional GPU limitations must be non-empty strings")
    local_path = (
        "l1_and_local_queue_evidence_unavailable"
        if hardware.l1_cache_enabled
        else "local_queue_evidence_unavailable"
    )
    common = additional + (
        local_path,
        "noncoalesced_transaction_geometry_unavailable",
        "workload_producer_independence_unavailable",
        "hardware_source_ceiling_evidence_unavailable",
        "request_frontend_capacity_evidence_unavailable",
    )
    if hardware.dram_arbitration_mode == "shared_directional_capacity":
        return common + (
            "dram_direction_switch_schedule_unavailable",
            "dram_turnaround_unmodeled",
        )
    if hardware.dram_arbitration_mode == "independent_directional_capacity":
        return common
    if hardware.dram_arbitration_mode == "shared_common_capacity":
        return common + (
            "write_capacity_evidence_unavailable",
            "dram_directional_capacity_and_arbitration_unavailable",
        )
    raise ValueError(
        "GPU hardware has no supported DRAM arbitration capability")


def npu_launch_floor_contract(
        hardware, launch_task_kind: str) -> Tuple[bool, Tuple[str, ...]]:
    """Return minimum task-duration evidence status for one NPU task class."""

    if not hardware.aic_aiv_separated:
        raise ValueError(
            "NPU launch-floor contract requires separated AIC/AIV hardware")
    if launch_task_kind not in ("aic", "aiv", "mix_aiv", "mixed"):
        raise ValueError(
            "NPU launch-floor task kind must be 'aic', 'aiv', 'mix_aiv', "
            "or 'mixed'")
    available = hardware.has_task_duration_floor_for_task(
        launch_task_kind)
    if available:
        return True, ()
    return False, ("launch_task_duration_evidence_unavailable",)


def split_global_memory_paths(
        inp: PredictionInput, *, read_l2_bytes: float,
        read_dram_bytes: float, write_l2_bytes: float,
        write_dram_bytes: float,
        dependency_paths: Optional[Tuple[AnalyticalDependencyPath, ...]] = None
        ) -> PredictionInput:
    """Split the canonical global path into physical read and write streams."""

    import dataclasses

    if len(inp.memory_paths) != 1:
        raise ValueError("global memory split requires exactly one path")
    original = inp.memory_paths[0]
    if original.access_class_demands:
        raise ValueError(
            "global memory split cannot redistribute a path with typed "
            "memory access-class demand")
    if original.name not in ("global_memory", "mte2_read"):
        raise ValueError(
            f"cannot split unexpected memory path {original.name!r}")
    totals = {
        "read_l2_bytes": read_l2_bytes,
        "read_dram_bytes": read_dram_bytes,
        "write_l2_bytes": write_l2_bytes,
        "write_dram_bytes": write_dram_bytes,
    }
    invalid = {
        name: value for name, value in totals.items()
        if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0)
    }
    if invalid:
        raise ValueError(
            f"memory path split bytes must be finite and non-negative: {invalid}")
    original_l2 = inp.demand_per_resource["l2_cache"]
    original_dram = inp.demand_per_resource["dram"]
    if not math.isclose(
            read_l2_bytes + write_l2_bytes, original_l2,
            rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("read/write L2 bytes do not conserve global traffic")
    if not math.isclose(
            read_dram_bytes + write_dram_bytes, original_dram,
            rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("read/write DRAM bytes do not conserve global traffic")
    transaction_bytes_by_role = {
        node.resource_role: node.transaction_bytes
        for node in original.nodes
    }

    def make_path(name, l2_bytes, dram_bytes):
        if l2_bytes <= 0:
            if dram_bytes != 0:
                raise ValueError(
                    f"{name} has DRAM bytes without ingress bytes")
            return None
        if dram_bytes > l2_bytes:
            raise ValueError(f"{name} DRAM bytes cannot exceed L2 bytes")
        l2_transaction_bytes = transaction_bytes_by_role["l2_cache"]
        if l2_transaction_bytes is None:
            raise ValueError(
                "global GPU path split requires resolved L2 transactions")
        nodes = [MemoryNodeDemand(
            resource_role="l2_cache",
            transactions=l2_bytes / l2_transaction_bytes,
            bytes=l2_bytes,
            transaction_bytes=l2_transaction_bytes,
        )]
        if dram_bytes > 0:
            dram_transaction_bytes = transaction_bytes_by_role["dram"]
            if dram_transaction_bytes is None:
                raise ValueError(
                    "global GPU path split requires resolved DRAM transactions")
            nodes.append(MemoryNodeDemand(
                resource_role="dram",
                transactions=dram_bytes / dram_transaction_bytes,
                bytes=dram_bytes,
                transaction_bytes=dram_transaction_bytes,
            ))
        return MemoryPathDemand(
            name=name,
            access_kind=("read" if name.endswith("read") else "write"),
            initial_transactions=l2_bytes / l2_transaction_bytes,
            nodes=tuple(nodes),
            source=original.source,
            queues=original.queues,
        )

    paths = tuple(
        path for path in (
            make_path("global_read", read_l2_bytes, read_dram_bytes),
            make_path("global_write", write_l2_bytes, write_dram_bytes),
        )
        if path is not None
    )
    replacement = {"memory_paths": paths}
    if dependency_paths is not None:
        replacement["dependency_paths"] = dependency_paths
    return dataclasses.replace(inp, **replacement)


def append_npu_write_path(
        inp: PredictionInput, hardware, *,
        dram_write_bytes: Optional[float] = None,
        l2_write_bytes: Optional[float] = None,
        write_active_units: Optional[int] = None,
        prepend_aiv_mte2_ingress: bool = False) -> PredictionInput:
    """Attach a byte-only UB/MTE3 stream and proven L2/GM continuations."""

    import dataclasses

    if inp.mem_model != "npu_tile":
        raise ValueError("NPU write path requires mem_model='npu_tile'")
    if not isinstance(prepend_aiv_mte2_ingress, bool):
        raise TypeError("prepend_aiv_mte2_ingress must be boolean")
    if any(path.name == "mte3_write" for path in inp.memory_paths):
        raise ValueError("NPU MTE3 write path must be attached exactly once")
    write_bytes = float(inp.demand_per_resource["mte3"])
    if write_bytes <= 0:
        raise ValueError("NPU MTE3 write path requires positive write bytes")
    if dram_write_bytes is None:
        normalized_dram_write_bytes = write_bytes
    else:
        if (isinstance(dram_write_bytes, bool)
                or not isinstance(dram_write_bytes, (int, float))
                or not math.isfinite(dram_write_bytes)
                or dram_write_bytes < 0):
            raise ValueError(
                "NPU MTE3 DRAM write bytes must be finite and non-negative")
        normalized_dram_write_bytes = float(dram_write_bytes)
    if normalized_dram_write_bytes > write_bytes:
        raise ValueError(
            "NPU MTE3 DRAM continuation cannot exceed MTE3 source bytes")
    if l2_write_bytes is None:
        normalized_l2_write_bytes = 0.0
    else:
        if (isinstance(l2_write_bytes, bool)
                or not isinstance(l2_write_bytes, (int, float))
                or not math.isfinite(l2_write_bytes)
                or l2_write_bytes < 0):
            raise ValueError(
                "NPU MTE3 L2 write bytes must be finite and non-negative")
        normalized_l2_write_bytes = float(l2_write_bytes)
    if normalized_l2_write_bytes > write_bytes:
        raise ValueError(
            "NPU MTE3 L2 continuation cannot exceed MTE3 source bytes")
    if (normalized_l2_write_bytes > 0
            and normalized_dram_write_bytes > normalized_l2_write_bytes):
        raise ValueError(
            "NPU MTE3 DRAM continuation cannot exceed its proven L2 path")
    if hardware.kernel_memory_model != "npu_tile":
        raise ValueError("NPU write path requires npu_tile hardware")
    if write_active_units is not None:
        if (isinstance(write_active_units, bool)
                or not isinstance(write_active_units, int)):
            raise TypeError("NPU MTE3 active units must be an integer")
        normalized_write_active_units = write_active_units
    elif inp.launch_task_kind == "aiv":
        normalized_write_active_units = inp.active_units
    elif inp.launch_task_kind == "mixed":
        # Mixed CANN tasks express the launch grid in AIC block-dim units.  The
        # MTE3 producer is the corresponding AIV domain, so preserve the same
        # active fraction while converting to physical AIV cores.
        mixed_domain_units = hardware.execution_units_for_task("mixed")
        aiv_domain_units = hardware.execution_units_for_task("aiv")
        scaled_units = inp.active_units * aiv_domain_units
        if scaled_units % mixed_domain_units:
            raise ValueError(
                "mixed NPU task does not map to an integral AIV active count")
        normalized_write_active_units = scaled_units // mixed_domain_units
    elif inp.launch_task_kind == "aic":
        raise ValueError("AIC-only task cannot publish an AIV MTE3 write path")
    else:
        raise AssertionError(
            f"unhandled NPU write launch task: {inp.launch_task_kind!r}")
    mte3_units = hardware.resource_by_role("mte3").total_units
    if (normalized_write_active_units <= 0
            or normalized_write_active_units > mte3_units):
        raise ValueError(
            "NPU MTE3 active units exceed the physical AIV domain: "
            f"active={normalized_write_active_units}, physical={mte3_units}")
    nodes = [MemoryNodeDemand(
        resource_role="mte3",
        transactions=None,
        bytes=write_bytes,
        transaction_bytes=None,
    )]
    if normalized_l2_write_bytes > 0:
        nodes.append(MemoryNodeDemand(
            resource_role="l2_cache",
            transactions=None,
            bytes=normalized_l2_write_bytes,
            transaction_bytes=None,
        ))
    if normalized_dram_write_bytes > 0:
        nodes.append(MemoryNodeDemand(
            resource_role="dram",
            transactions=None,
            bytes=normalized_dram_write_bytes,
            transaction_bytes=None,
        ))
    path = MemoryPathDemand(
        name="mte3_write",
        access_kind="write",
        initial_transactions=None,
        nodes=tuple(nodes),
        source=MemorySourceDemand(
            active_units=normalized_write_active_units,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
    )
    flat = dict(inp.demand_per_resource)
    flat["l2_cache"] += normalized_l2_write_bytes
    flat["dram"] += normalized_dram_write_bytes
    memory_paths = (
        _npu_aiv_mte2_ingress_paths(inp, hardware)
        if prepend_aiv_mte2_ingress else inp.memory_paths)
    return dataclasses.replace(
        inp,
        demand_per_resource=flat,
        memory_paths=memory_paths + (path,),
    )


def _npu_aiv_mte2_ingress_paths(inp: PredictionInput, hardware):
    """Return memory paths with the source-defined AIV MTE2 ingress."""

    if (inp.mem_model != "npu_tile"
            or hardware.kernel_memory_model != "npu_tile"):
        raise ValueError("NPU AIV MTE2 ingress requires npu_tile memory")
    if inp.launch_task_kind != "aiv":
        raise ValueError("NPU AIV MTE2 ingress requires an AIV-only task")
    matches = tuple(
        (index, path)
        for index, path in enumerate(inp.memory_paths)
        if path.name == "mte2_read")
    if not matches:
        if inp.demand_per_resource["l2_cache"] != 0.0:
            raise ValueError(
                "positive NPU AIV reads require an mte2_read path")
        return inp.memory_paths
    if len(matches) != 1:
        raise ValueError(
            "NPU AIV MTE2 ingress requires exactly one mte2_read path")
    path_index, original = matches[0]
    if original.access_kind != "read":
        raise ValueError("mte2_read must be a read path")
    if original.nodes[0].resource_role != "l2_cache":
        raise ValueError(
            "NPU AIV MTE2 ingress requires an L2-first continuation path")
    if any(node.resource_role == "mte2" for node in original.nodes):
        raise ValueError("NPU AIV MTE2 ingress must be attached exactly once")
    mte2_units = hardware.resource_by_role("mte2").total_units
    if inp.active_units > mte2_units:
        raise ValueError(
            "NPU MTE2 active units exceed the physical AIV domain: "
            f"active={inp.active_units}, physical={mte2_units}")
    ingress = MemoryNodeDemand(
        resource_role="mte2",
        transactions=None,
        bytes=original.nodes[0].bytes,
        transaction_bytes=None,
    )
    paths = list(inp.memory_paths)
    paths[path_index] = dataclasses.replace(
        original, nodes=(ingress,) + original.nodes)
    return tuple(paths)


def prepend_npu_aiv_mte2_ingress(
        inp: PredictionInput, hardware) -> PredictionInput:
    """Prepend the physical per-AIV MTE2 endpoint to an NPU read path."""

    import dataclasses

    paths = _npu_aiv_mte2_ingress_paths(inp, hardware)
    if paths is inp.memory_paths:
        return inp
    return dataclasses.replace(inp, memory_paths=paths)


def attach_uniform_pipeline(
        inp: PredictionInput, hardware, *, task_count: int,
        stages: Tuple[Tuple[str, str, bool], ...], buffer_slots: int
        ) -> PredictionInput:
    """Attach a direct-grid homogeneous worker plan and remove old tail eta."""

    import dataclasses

    if isinstance(task_count, bool) or not isinstance(task_count, int):
        raise TypeError("pipeline task_count must be an integer")
    if task_count <= 0:
        raise ValueError("pipeline task_count must be positive")
    if isinstance(buffer_slots, bool) or not isinstance(buffer_slots, int):
        raise TypeError("pipeline buffer_slots must be an integer")
    if buffer_slots <= 0:
        raise ValueError("pipeline buffer_slots must be positive")
    if inp.pipeline_plan is not None:
        raise ValueError("pipeline plan must be attached exactly once")
    if not isinstance(stages, tuple) or not stages:
        raise TypeError("pipeline stages must be a non-empty tuple")
    worker_count = min(task_count, inp.active_units)
    fraction = 1.0 / task_count
    tasks = tuple(
        PipelineTaskDemand(
            name=f"worker{worker}_tiles",
            worker=worker,
            stages=tuple(
                bind_pipeline_stage_template(
                    name=stage_name,
                    service=service,
                    work_fraction=fraction,
                    include_first_request_latency=include_first,
                )
                for stage_name, service, include_first in stages
            ),
            repeat_count=(task_count - worker + worker_count - 1)
            // worker_count,
        )
        for worker in range(worker_count)
    )
    buffers = ()
    stage_names = tuple(stage[0] for stage in stages)
    if "load" in stage_names:
        load_index = stage_names.index("load")
        if load_index + 1 >= len(stage_names):
            if len(stage_names) != 1:
                raise ValueError(
                    "pipeline load stage must have a consumer")
        else:
            buffers += (PipelineBuffer(
                name="load_compute_buffer",
                producer_stage="load",
                consumer_stage=stage_names[load_index + 1],
                slots=buffer_slots,
                scope="worker",
            ),)
    if "store" in stage_names:
        store_index = stage_names.index("store")
        if store_index == 0:
            if len(stage_names) != 1:
                raise ValueError(
                    "pipeline store stage must have a producer")
        else:
            buffers += (PipelineBuffer(
                name="compute_store_buffer",
                producer_stage=stage_names[store_index - 1],
                consumer_stage="store",
                slots=buffer_slots,
                scope="worker",
            ),)

    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=tasks,
        buffers=buffers,
    )
    return attach_pipeline_work_plan(inp, hardware, plan)


def attach_pipeline_work_plan(
        inp: PredictionInput, hardware,
        plan: PipelineWorkPlan, *,
        promote_npu_partial: bool = False) -> PredictionInput:
    """Attach an explicit plan and remove only the legacy tail-wave factor."""

    import dataclasses

    if not isinstance(plan, PipelineWorkPlan):
        raise TypeError("plan must be PipelineWorkPlan")
    if inp.pipeline_plan is not None:
        raise ValueError("pipeline plan must be attached exactly once")
    if not isinstance(promote_npu_partial, bool):
        raise TypeError("promote_npu_partial must be boolean")
    if promote_npu_partial:
        if (inp.mem_model != "npu_tile"
                or inp.support_level != "aggregate_only"):
            raise ValueError(
                "NPU partial promotion requires an aggregate-only npu_tile "
                "input")
    elif inp.mem_model == "npu_tile":
        raise ValueError(
            "npu_tile pipeline attachment requires explicit partial-support "
            "promotion by its source-owning predictor")
    old_wave_efficiency = inp.wave_efficiency
    if not 0 < old_wave_efficiency <= 1:
        raise ValueError("invalid legacy wave efficiency")
    compute_demands = tuple(
        dataclasses.replace(
            demand,
            schedule_efficiency=min(
                1.0,
                demand.schedule_efficiency / old_wave_efficiency),
        )
        for demand in inp.compute_demands
    )
    support_level = "partial" if promote_npu_partial else inp.support_level
    return dataclasses.replace(
        inp,
        compute_demands=compute_demands,
        pipeline_plan=plan,
        wave_efficiency=1.0,
        support_level=support_level,
    )


def attach_attention_pipeline_envelope(
        inp: PredictionInput, hardware,
        envelope: AttentionPipelineEnvelopeDemand) -> PredictionInput:
    """Attach a constant-size attention envelope and remove tail scaling."""

    import dataclasses

    if not isinstance(envelope, AttentionPipelineEnvelopeDemand):
        raise TypeError("envelope must be AttentionPipelineEnvelopeDemand")
    if inp.pipeline_plan is not None:
        raise ValueError("pipeline plan must be attached exactly once")
    if inp.mem_model == "npu_tile":
        raise ValueError("attention pipeline envelope is GPU-only")
    old_wave_efficiency = inp.wave_efficiency
    if not 0 < old_wave_efficiency <= 1:
        raise ValueError("invalid legacy wave efficiency")
    compute_demands = tuple(
        dataclasses.replace(
            demand,
            schedule_efficiency=min(
                1.0,
                demand.schedule_efficiency / old_wave_efficiency),
        )
        for demand in inp.compute_demands
    )
    return dataclasses.replace(
        inp,
        compute_demands=compute_demands,
        pipeline_plan=envelope,
        wave_efficiency=1.0,
    )


def pipeline_compute_service(
        inp: PredictionInput, hardware,
        roles: Optional[Tuple[str, ...]] = None) -> str:
    """Return one declared service covering all positive compute demands."""

    positive_roles = tuple(
        demand.resource_role for demand in inp.compute_demands
        if demand.operations > 0)
    if roles is None:
        arithmetic_roles = tuple(
            role for role in positive_roles if role != "warp_issue")
        roles = arithmetic_roles if arithmetic_roles else positive_roles
    elif (not isinstance(roles, tuple)
          or not roles
          or any(role not in positive_roles for role in roles)):
        raise ValueError(
            "requested pipeline compute roles must be a non-empty subset of "
            f"positive demands: requested={roles}, positive={positive_roles}")
    if not roles:
        raise ValueError("pipeline requires positive compute work")
    resource_names = {
        hardware.resource_by_role(role).name for role in roles
    }
    covering_groups = tuple(
        group.name for group in hardware.resource_groups
        if resource_names.issubset(set(group.members)))
    if len(covering_groups) == 1:
        return f"group:{covering_groups[0]}"
    if len(roles) == 1:
        return f"compute:{roles[0]}"
    if not covering_groups:
        raise NotImplementedError(
            "pipeline cannot serialize or overlap compute roles without a "
            f"shared hardware group: {roles}")
    raise ValueError(
        "compute demands are covered by multiple ambiguous resource groups: "
        f"{covering_groups}")


def pipeline_compute_stages(
        inp: PredictionInput, hardware
        ) -> Tuple[Tuple[str, str, bool], ...]:
    """Partition positive compute work into ordered physical group services."""

    positive_roles = tuple(
        demand.resource_role for demand in inp.compute_demands
        if demand.operations > 0 and demand.resource_role != "warp_issue")
    if not positive_roles:
        positive_roles = tuple(
            demand.resource_role for demand in inp.compute_demands
            if demand.operations > 0)
    if not positive_roles:
        raise ValueError("pipeline requires positive compute work")

    services = []
    for role in positive_roles:
        resource_name = hardware.resource_by_role(role).name
        covering_groups = tuple(
            group.name for group in hardware.resource_groups
            if resource_name in group.members)
        if len(covering_groups) == 0:
            service = f"compute:{role}"
        elif len(covering_groups) == 1:
            service = f"group:{covering_groups[0]}"
        else:
            raise ValueError(
                "compute resource belongs to multiple ambiguous groups: "
                f"role={role!r}, groups={covering_groups}")
        if service not in services:
            services.append(service)

    if len(services) == 1:
        return (("compute", services[0], False),)
    return tuple(
        (f"compute_{index}", service, False)
        for index, service in enumerate(services))


def build_forward_demand(l2_total: float, dram_bytes: float,
                         is_non_tensor: bool, flops: Dict[str, float]) -> Dict[str, float]:
    """Build the complete resource demand used by the physical formula."""

    compute_roles = {
        "compute_tensor", "compute_vector", "compute_reduction",
        "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
        "scalar_aiv",
    }
    unknown_roles = set(flops) - compute_roles
    if unknown_roles:
        raise ValueError(f"unknown compute demand roles: {sorted(unknown_roles)}")
    demand = {
        "l1_cache": l2_total if is_non_tensor else 0.0,
        "smem": 0.0 if is_non_tensor else l2_total * 2.0,
        "xbar": l2_total,
        "l2_cache": l2_total,
        "dram": dram_bytes,
        "mte3": 0.0,
        **{role: 0.0 for role in compute_roles},
    }
    demand.update({name: float(value) for name, value in flops.items()})
    return demand


def validated_l2_miss_rate(dram_bytes: float, l2_bytes: float) -> float:
    """Return the physical DRAM/L2 ratio without hiding inconsistent traffic."""

    if dram_bytes < 0 or l2_bytes < 0:
        raise ValueError("DRAM and L2 traffic cannot be negative")
    if l2_bytes == 0:
        if dram_bytes != 0:
            raise ValueError("positive DRAM traffic requires positive L2 traffic")
        return 0.0
    rate = dram_bytes / l2_bytes
    if not 0.0 <= rate <= 1.0:
        raise ValueError(
            f"DRAM/L2 traffic ratio must be within [0, 1], got {rate}")
    return rate


def causal_attention_pairs(
        q_lens: Tuple[int, ...], kv_lens: Tuple[int, ...], *,
        capacity: Optional[int] = None) -> int:
    """Count source-visible causal query/key pairs without shape padding."""

    if len(q_lens) != len(kv_lens) or not q_lens:
        raise ValueError("causal attention lengths must be non-empty and align")
    if capacity is not None and (
            isinstance(capacity, bool) or not isinstance(capacity, int)
            or capacity <= 0):
        raise ValueError("causal attention capacity must be positive")
    total = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        if (isinstance(q_len, bool) or not isinstance(q_len, int)
                or isinstance(kv_len, bool) or not isinstance(kv_len, int)
                or q_len <= 0 or kv_len < q_len):
            raise ValueError(
                "causal attention requires positive integer lengths and "
                "kv_len >= q_len")
        first_visible = kv_len - q_len + 1
        last_visible = kv_len
        if capacity is None or capacity >= last_visible:
            total += q_len * (first_visible + last_visible) // 2
            continue
        uncapped_terms = max(0, min(q_len, capacity - first_visible + 1))
        if uncapped_terms:
            total += uncapped_terms * (
                2 * first_visible + uncapped_terms - 1) // 2
        total += (q_len - uncapped_terms) * capacity
    if total <= 0:
        raise ValueError("causal attention pair count must be positive")
    return total


def launch_wave_geometry(kernel: KernelSpec, hardware, *,
                         execution_units: Optional[int] = None
                         ) -> Tuple[float, float, int]:
    """Return fixed resource occupancy, tail-wave efficiency and active units.

    ``smem``, registers, CTA residency and resident warps are deliberately
    absent.  A finite grid still exposes a physical tail because workgroups
    beyond the active units require a later wave.
    """

    launch = kernel.launch
    if execution_units is None:
        unit_count = hardware.num_compute_units
    else:
        if (isinstance(execution_units, bool)
                or not isinstance(execution_units, int)):
            raise TypeError("execution_units must be an integer")
        if execution_units <= 0:
            raise ValueError("execution_units must be positive")
        unit_count = execution_units
    wave_size = min(launch.grid_size, unit_count)
    if wave_size <= 0:
        raise ValueError("launch produced a non-positive wave size")
    waves = launch.grid_size / wave_size
    wave_efficiency = waves / math.ceil(waves)
    active_units = min(launch.grid_size, unit_count)
    if hardware.kernel_memory_model not in (
            "gpu_mshr", "npu_tile", "pim_bank", "systolic_scratchpad"):
        raise ValueError(
            "unknown kernel_memory_model for launch resources: "
            f"{hardware.kernel_memory_model!r}")
    return (
        FULL_RESOURCE_OCCUPANCY,
        wave_efficiency,
        active_units,
    )


def build_aggregate_gpu_input(
        kernel: KernelSpec, hardware, demand: Dict[str, float], *,
        write_l2_bytes: float, write_dram_bytes: float,
        limitations: Tuple[str, ...]) -> PredictionInput:
    """Build a source-work aggregate for an uncalibrated GPU kernel.

    This boundary is intentionally limited to static semantic work and tensor
    traffic.  It does not infer residency, pipeline overlap, or a workload
    timing correction when the retained source does not expose those facts.
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise NotImplementedError(
            f"{kernel.kind.value}/{kernel.variant} currently has only a "
            "GPU analytical execution contract")
    if not isinstance(limitations, tuple) or any(
            not isinstance(value, str) or not value for value in limitations):
        raise TypeError("aggregate GPU limitations must be non-empty strings")
    l2_total = float(demand["l2_cache"])
    dram_total = float(demand["dram"])
    for name, value, total in (
            ("write_l2_bytes", write_l2_bytes, l2_total),
            ("write_dram_bytes", write_dram_bytes, dram_total)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0 or value > total):
            raise ValueError(
                f"{name} must be finite and within its total traffic")
    eta, wave_efficiency, active_units = launch_wave_geometry(
        kernel, hardware)
    inp = PredictionInput(
        demand_per_resource=dict(demand),
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=kernel.variant,
        mem_model=hardware.kernel_memory_model,
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind="generic",
        dram_capacity_multiplier=hardware.dram_protocol_efficiency_l1cached,
        include_dispatch_overhead=True,
        support_level="aggregate_only",
        limitations=gpu_partial_limitations(hardware, *limitations),
    )
    producers, requests_per_producer = gpu_memory_source(hardware)
    structured = attach_structured_contract(
        inp,
        hardware,
        transaction_bytes=None,
        path_name="global_memory",
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=None,
    )
    return split_global_memory_paths(
        structured,
        read_l2_bytes=l2_total - float(write_l2_bytes),
        read_dram_bytes=dram_total - float(write_dram_bytes),
        write_l2_bytes=float(write_l2_bytes),
        write_dram_bytes=float(write_dram_bytes),
    )


def roofline_ns(demand: Dict[str, float], hardware, compulsory_bytes: float) -> float:
    compute_seconds = 0.0
    for role in (
            "compute_tensor", "compute_vector", "compute_reduction",
            "compute_fma", "compute_sfu", "compute_alu"):
        amount = float(demand[role])
        if amount <= 0:
            continue
        if role not in hardware._role_map:
            raise ValueError(
                f"hardware has no {role!r} resource for positive demand {amount}")
        capacity = hardware.resource_by_role(role).capacity
        if capacity <= 0:
            raise ValueError(f"hardware {role!r} capacity must be positive")
        compute_seconds = max(compute_seconds, amount / capacity)
    dram_capacity = hardware.resource_by_role("dram").capacity
    if compulsory_bytes < 0 or dram_capacity <= 0:
        raise ValueError(
            "roofline bytes cannot be negative and DRAM capacity must be positive")
    memory_seconds = compulsory_bytes / dram_capacity
    return max(compute_seconds, memory_seconds) * 1e9


def _with_graph_regime(inp, kernel):
    """Tag the input with whether this launch is a replayed graph node."""

    replayed = graph_replayed_launch(kernel)
    if replayed == inp.graph_replayed:
        return inp
    return dataclasses.replace(inp, graph_replayed=replayed)
