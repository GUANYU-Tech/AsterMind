"""Evidence-preserving reports for single-kernel mechanism validation.

This module reads production predictor inputs and diagnostics.  It does not
reimplement or adjust a predictor formula, and measured duration is retained
only as an evaluation target.
"""

from __future__ import annotations

from evaluation.json_shape import file_sha256

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from evaluation.kernel.dataset import (
    ATTENTION_OP_TYPES,
    MOE_OP_TYPES,
    VECTOR_OP_TYPES,
    get_hardware_config_path,
    get_kernel_dataset_path,
    load_kernel_dataset,
    to_float,
)
from evaluation.kernel.oracle_observation import (
    single_counterfactual_inputs,
)
from src.kernel.kernel_base import create_predictor
from evaluation.kernel.evaluation_runner import _process_op_star


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "experiment_logs/kernel/mechanism_validation"
)
HARDWARE = (
    ("nvidia_a800", "NVIDIA A800"),
    ("metax_c500", "MetaX C500"),
    ("ascend_910b", "Ascend 910B"),
)
DATASET_FAMILIES = ("gemm", "flash_attn", "fused_moe", "vector")
FAMILY_LABELS = ("GEMM", "FlashAttention", "FusedMoE", "Vector")
VECTOR_TYPES = VECTOR_OP_TYPES
DEMAND_ROLES = (
    "compute_tensor", "compute_vector", "compute_reduction", "compute_fma",
    "compute_sfu", "compute_alu", "warp_issue", "l1_cache", "smem",
    "xbar", "l2_cache", "dram", "mte3",
)
TIME_ROLES = (
    "compute_tensor", "compute_vector", "compute_reduction", "compute_fma",
    "compute_sfu", "compute_alu", "warp_issue", "l1_cache", "xbar",
    "l2_cache", "dram", "mte3", "scalar_aiv",
)
NVIDIA_L2_SECTOR_BYTES = 32.0
DIRECTIONAL_TRAFFIC_COMPONENTS = (
    "traffic:l2c_source_read",
    "traffic:downstream_global_read",
    "traffic:downstream_global_write",
    "traffic:xbar_read",
    "traffic:xbar_write",
    "traffic:l2_source_read",
    "traffic:l2_source_write",
    "traffic:dram_read_in_window",
    "traffic:dram_writeback_in_window",
    "traffic:aic_mte2_gm_to_l1",
    "traffic:aiv_mte2_gm_to_ub",
    "traffic:aiv_mte3_ub_to_gm",
    "traffic:dram_read_derived",
)
PROFILER_EVIDENCE_KINDS = frozenset({"profiler", "profiler_derived"})
BUCKET_SCHEMA_VERSION = "physical_bucket_v1"
TASK_AGGREGATION_CONTRACT = "serial_sum_of_forward_kernel_predictions_v2"
BUCKET_MAPE_LIMIT_PCT = 15.0
CASE_APE_LIMIT_PCT = 30.0
SUPPORT_LEVELS = (
    "complete",
    "partial",
    "aggregate_only",
    "compatibility",
)
COMPARISON_STATUSES = (
    "comparable",
    "not_comparable",
    "unobserved",
)
CURRENT_DATASET_COMPARISON_REASON = (
    "capture_software_source_and_binary_identity_not_recorded"
)
PHYSICAL_BUCKET_DISCRETE_CONTROLS = (
    "aiv_mixed",
    "group_size_m",
    "operand_replay_a",
    "operand_replay_b",
    "expert_parallel_size",
    "cann_tiling_key",
)
SEVERE_DURATION_FIELDS = (
    "result_scope",
    "hardware_id",
    "family",
    "case_id",
    "kernel_ordinal",
    "kernel_kind",
    "variant",
    "kernel_count",
    "timing_source",
    "predicted_ns",
    "observed_ns",
    "absolute_duration_error_ns",
    "signed_error_pct",
    "absolute_error_pct",
    "physical_bucket_id",
    "comparison_status",
    "comparison_reason",
)
CASE_IMPACT_FIELDS = (
    "impact_rank",
    "result_scope",
    "hardware_id",
    "family",
    "case_id",
    "kernel_ordinal",
    "kernel_kind",
    "variant_or_execution_graph",
    "kernel_count",
    "timing_source",
    "measured_duration_field",
    "measured_ns",
    "predicted_ns",
    "absolute_duration_error_ns",
    "signed_error_pct",
    "absolute_error_pct",
    "case_ape_gate_passed",
    "physical_bucket_id",
    "bucket_mape_pct",
    "bucket_mape_gate_passed",
    "bucket_max_ape_pct",
    "bucket_max_ape_gate_passed",
    "support_level",
    "comparison_status",
    "comparison_reason",
)
C500_SILU_MECHANISM_FIELDS = (
    "result_scope",
    "case_id",
    "tokens",
    "intermediate",
    "production_duration_ns",
    "timing_source",
    "predicted_duration_ns",
    "predicted_over_production",
    "absolute_error_pct",
    "grid_size",
    "waves",
    "predicted_instruction_lower_bound",
    "observed_total_instructions",
    "instruction_predicted_over_observed",
    "instruction_comparison_status",
    "instruction_comparison_reason",
    "predicted_global_bytes",
    "observed_global_bytes",
    "traffic_predicted_over_observed",
    "instrumented_duration_ns",
    "instrumented_core_clock_ghz",
    "instrumented_ap_active_cycles",
    "instrumented_mean_active_aps",
    "instrumented_instruction_per_active_cycle",
    "instrumented_average_wave_life_cycles",
    "instrumented_dnoc_read_latency_cycles",
    "instrumented_vls_pipeline_stall_cycles",
    "instrumented_vls_wdata_stall_cycles",
    "timing_scope_note",
)
BEFORE_AFTER_CASE_FIELDS = (
    "result_scope",
    "hardware_id",
    "family",
    "case_id",
    "kernel_ordinal",
    "kernel_kind",
    "before_variant_or_execution_graph",
    "variant_or_execution_graph",
    "before_model_identity",
    "after_model_identity",
    "model_identity_changed",
    "timing_source",
    "measured_ns",
    "before_predicted_ns",
    "after_predicted_ns",
    "prediction_change_ns",
    "before_signed_error_pct",
    "after_signed_error_pct",
    "before_absolute_error_pct",
    "after_absolute_error_pct",
    "ape_change_pct_points",
    "before_absolute_duration_error_ns",
    "after_absolute_duration_error_ns",
    "absolute_duration_error_change_ns",
    "before_case_gate_passed",
    "after_case_gate_passed",
    "before_support_level",
    "after_support_level",
    "support_level_changed",
    "bucket_schema",
    "physical_bucket_id",
    "physical_bucket_short_id",
    "bucket_dimensions",
    "bucket_basis",
    "comparison_status",
    "comparison_reason",
)
BEFORE_AFTER_BUCKET_FIELDS = (
    "result_scope",
    "hardware_id",
    "family",
    "physical_bucket_id",
    "physical_bucket_short_id",
    "bucket_schema",
    "bucket_dimensions",
    "bucket_basis",
    "row_count",
    "comparable_count",
    "not_comparable_count",
    "unobserved_count",
    "comparison_coverage_pct",
    "comparison_coverage_gate_passed",
    "before_mape_pct",
    "after_mape_pct",
    "mape_change_pct_points",
    "before_mdape_pct",
    "after_mdape_pct",
    "before_p90_ape_pct",
    "after_p90_ape_pct",
    "before_p99_ape_pct",
    "after_p99_ape_pct",
    "before_max_ape_pct",
    "after_max_ape_pct",
    "max_ape_change_pct_points",
    "before_mean_signed_error_pct",
    "after_mean_signed_error_pct",
    "before_median_signed_error_pct",
    "after_median_signed_error_pct",
    "before_case_violation_count",
    "after_case_violation_count",
    "before_diagnostic_mape_gate_passed",
    "after_diagnostic_mape_gate_passed",
    "before_diagnostic_max_ape_gate_passed",
    "after_diagnostic_max_ape_gate_passed",
    "before_diagnostic_accuracy_gate_passed",
    "after_diagnostic_accuracy_gate_passed",
    "before_mape_gate_passed",
    "after_mape_gate_passed",
    "before_max_ape_gate_passed",
    "after_max_ape_gate_passed",
    "before_accuracy_gate_passed",
    "after_accuracy_gate_passed",
)
BEFORE_AFTER_SUMMARY_FIELDS = (
    "result_scope",
    "hardware_id",
    "family",
    "row_count",
    "before_mape_pct",
    "after_mape_pct",
    "mape_change_pct_points",
    "before_mdape_pct",
    "after_mdape_pct",
    "before_p90_ape_pct",
    "after_p90_ape_pct",
    "before_p99_ape_pct",
    "after_p99_ape_pct",
    "before_max_ape_pct",
    "after_max_ape_pct",
    "before_mean_signed_error_pct",
    "after_mean_signed_error_pct",
    "before_median_signed_error_pct",
    "after_median_signed_error_pct",
    "before_case_pass_count",
    "after_case_pass_count",
    "before_case_violation_count",
    "after_case_violation_count",
)
SEMANTICALLY_EXCLUDED_COMPONENTS = frozenset({
    ("ascend_910b", "demand:compute_tensor"),
    ("ascend_910b", "demand:compute_vector"),
    ("ascend_910b", "demand:compute_reduction"),
    ("ascend_910b", "demand:compute_fma"),
    ("ascend_910b", "demand:compute_sfu"),
    ("ascend_910b", "demand:compute_alu"),
    ("ascend_910b", "demand:warp_issue"),
    # mcProfiler's l2c_memacs is an upstream read-side request count, whereas
    # the production xbar/L2 roles include both read and write nodes.  The
    # downstream all_memacs counter is read+write at a different transaction
    # boundary.  Neither is a like-for-like observation of the aggregate
    # xbar/L2 demand, so compare the directional memory paths instead.
    ("metax_c500", "demand:xbar"),
    ("metax_c500", "demand:l2_cache"),
    # The native directional counters are retained separately below.  These
    # aggregate rows would otherwise hide read/write mistakes and, for L2,
    # compare against a service envelope rather than one counter interface.
    ("nvidia_a800", "demand:l2_cache"),
    ("nvidia_a800", "demand:dram"),
    # NPU MTE2 reads, MTE3 writes, and DRAM-read misses are distinct paths.
    # Flat xbar/L2/DRAM/MTE3 replacement loses that direction and can inject a
    # positive GPU-shaped xbar role which the npu_tile contract intentionally
    # leaves at zero.
    ("ascend_910b", "demand:xbar"),
    ("ascend_910b", "demand:l2_cache"),
    ("ascend_910b", "demand:dram"),
    ("ascend_910b", "demand:mte3"),
})


def _is_semantically_comparable_component(
        hardware_id: str, family: str, variant: str,
        component: str, profiler_op_type: str | None) -> bool:
    if (hardware_id, component) in SEMANTICALLY_EXCLUDED_COMPONENTS:
        return False
    if hardware_id == "ascend_910b":
        if profiler_op_type not in ("cube", "vector", "mix"):
            raise ValueError(
                "Ascend semantic comparison requires a known profiler "
                f"Op Type, got {profiler_op_type!r}")
        if profiler_op_type == "mix" and component == "demand:dram":
            # Historical canonical mixed-domain records applied one mean AIC
            # hit rate to the combined AIC+AIV request volume.  The per-row
            # join needed to reconstruct DRAM misses was not retained, so the
            # resulting aggregate is not a valid comparison target.
            return False
        if (family == "FusedMoE"
                and variant in ("w1", "w2")
                and component in ("demand:l2_cache", "demand:dram")):
            # Forward now exposes separate GroupedMatmul read and output
            # interfaces.  The generic aggregate component does not have that
            # split and therefore is not the matching validation boundary.
            return False
    return True


def _family(op_type: str) -> str:
    if op_type == "GEMM":
        return "GEMM"
    elif op_type in ATTENTION_OP_TYPES:
        return "FlashAttention"
    elif op_type in MOE_OP_TYPES:
        return "FusedMoE"
    elif op_type in VECTOR_TYPES:
        return "Vector"
    else:
        raise ValueError(f"unsupported mechanism-validation op type: {op_type!r}")


def _canonical_capture_comparison_evidence(
        hardware_id: str) -> Mapping[str, str]:
    """Return the fail-closed status of current timing and counter captures.

    All three canonical hardware datasets preserve a timer source and exact
    kernel ordinal, and most preserve profiler counters.  None records the
    capture-time runtime, source snapshot, runner binary, and operator binary
    identities needed to bind those observations to the current production
    Forward.  A missing identity is ``unobserved``; it is not evidence of a
    version mismatch and must not be called ``not_comparable``.
    """

    if hardware_id not in {item[0] for item in HARDWARE}:
        raise ValueError(
            f"unknown duration-comparison hardware: {hardware_id!r}")
    return {
        "comparison_status": "unobserved",
        "comparison_reason": CURRENT_DATASET_COMPARISON_REASON,
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(name): _json_value(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if hasattr(value, "value"):
        return _json_value(value.value)
    raise TypeError(
        f"mechanism report cannot serialize {type(value).__name__}")


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _lfs_oids() -> Mapping[str, str]:
    output = _git_output("lfs", "ls-files", "-l")
    identities = {}
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"unexpected git-lfs identity line: {line!r}")
        oid, marker, path = parts
        if marker not in ("*", "-"):
            raise ValueError(f"unexpected git-lfs worktree marker: {line!r}")
        if path.startswith("kernel_dataset/") or path.startswith("launch_data/"):
            identities[path] = oid
    return identities


def _tiling_summary(tiling) -> Mapping[str, Any] | None:
    if tiling is None:
        return None
    return {
        "tile_shape": dict(tiling.tile_shape),
        "pipeline_depth": tiling.pipeline_depth,
        "split_k": tiling.split_k,
        "num_kv_splits": tiling.num_kv_splits,
    }


def _launch_summary(launch) -> Mapping[str, Any]:
    return {
        "algorithm": launch.algorithm,
        "grid": list(launch.grid),
        "grid_size": launch.grid_size,
        "block": list(launch.block),
        "block_size": launch.block_size,
        "tiling": _tiling_summary(launch.tiling),
        "cluster": None if launch.cluster is None else list(launch.cluster),
        "parameters": dict(launch.parameters),
        "execution_variant": launch.execution_variant,
        "source_program_type": (
            None if launch.source_program is None
            else type(launch.source_program).__name__),
        "source_program": (
            None if launch.source_program is None
            else _json_value(launch.source_program)),
    }


def _compute_demand_summary(prediction_input) -> list[Mapping[str, Any]]:
    result = []
    for demand in prediction_input.compute_demands:
        services = []
        for service in demand.service_demands:
            services.append({
                "service": service.service,
                "operations": service.operations,
                "serial_invocations": service.serial_invocations,
                "worker_count": len(service.operations_by_worker),
                "max_worker_operations": (
                    max(service.operations_by_worker)
                    if service.operations_by_worker else None),
            })
        exact_requests = []
        for request in demand.exact_request_demands:
            request_counts = tuple(
                sum(count for _, count in worker_runs)
                for worker_runs in request.request_size_runs)
            exact_requests.append({
                "service": request.service,
                "operations": request.operations,
                "worker_count": request.worker_count,
                "request_count": sum(request_counts),
                "max_worker_request_count": max(request_counts),
            })
        result.append({
            "resource_role": demand.resource_role,
            "operations": demand.operations,
            "shape_efficiency": demand.shape_efficiency,
            "schedule_efficiency": demand.schedule_efficiency,
            "service_demands": services,
            "exact_request_demands": exact_requests,
        })
    return result


def _memory_path_summary(prediction_input) -> list[Mapping[str, Any]]:
    result = []
    for path in prediction_input.memory_paths:
        result.append({
            "name": path.name,
            "access_kind": path.access_kind,
            "rate_unit": path.rate_unit,
            "initial_transactions": path.initial_transactions,
            "nodes": [
                {
                    "resource_role": node.resource_role,
                    "transactions": node.transactions,
                    "bytes": node.bytes,
                    "transaction_bytes": node.transaction_bytes,
                }
                for node in path.nodes
            ],
            "source": {
                "active_units": path.source.active_units,
                "producers_per_unit": path.source.producers_per_unit,
                "independent_requests_per_producer": (
                    path.source.independent_requests_per_producer),
                "frontend_rate_per_unit_s": path.source.frontend_rate_per_unit_s,
                "workload_slots_per_unit": path.source.workload_slots_per_unit,
            },
            "queues": [
                {
                    "name": queue.name,
                    "allocations_per_initial_request": (
                        queue.allocations_per_initial_request),
                    "lifetime_weights": [list(item) for item in queue.lifetime_weights],
                }
                for queue in path.queues
            ],
            "access_classes": [_json_value(value) for value in path.access_class_demands],
        })
    return result


def _pipeline_input_summary(prediction_input) -> Mapping[str, Any] | None:
    plan = prediction_input.pipeline_plan
    if plan is None:
        return None
    result = {"type": type(plan).__name__}
    for name in ("worker_count", "active_aic_units", "active_aiv_units"):
        result[name] = getattr(plan, name) if hasattr(plan, name) else None
    for name in ("resources", "tasks", "buffers", "aic_groups"):
        if hasattr(plan, name):
            value = getattr(plan, name)
            result[f"{name}_encoded_count"] = len(value)
        else:
            result[f"{name}_encoded_count"] = None
    return result


def _prediction_input_summary(prediction_input) -> Mapping[str, Any]:
    return {
        "demand_per_resource": dict(prediction_input.demand_per_resource),
        "eta": prediction_input.eta,
        "wave_efficiency": prediction_input.wave_efficiency,
        "active_units": prediction_input.active_units,
        "sm_freq_hz": prediction_input.sm_freq_hz,
        "dram_freq_hz": prediction_input.dram_freq_hz,
        "grid_size": prediction_input.grid_size,
        "block_size": prediction_input.block_size,
        "variant": prediction_input.variant,
        "mem_model": prediction_input.mem_model,
        "compute_demands": _compute_demand_summary(prediction_input),
        "memory_paths": _memory_path_summary(prediction_input),
        "pipeline_plan": _pipeline_input_summary(prediction_input),
        "phase_ledger_present": prediction_input.phase_ledger is not None,
        "dram_capacity_multiplier": prediction_input.dram_capacity_multiplier,
        "include_dispatch_overhead": prediction_input.include_dispatch_overhead,
        "support_level": prediction_input.support_level,
        "limitations": list(prediction_input.limitations),
        "resource_time_overrides_ns": dict(
            prediction_input.resource_time_overrides_ns),
        "launch_task_kind": prediction_input.launch_task_kind,
        "task_duration_floor_contract": (
            prediction_input.task_duration_floor_contract),
        "execution_time_floor_ns": prediction_input.execution_time_floor_ns,
    }


def _prediction_summary(prediction, hardware) -> Mapping[str, Any]:
    diagnostics = prediction.diagnostics
    if diagnostics is None:
        raise RuntimeError("mechanism validation requires Forward diagnostics")
    resource_time_by_role_ns = {}
    for role in TIME_ROLES:
        if role in hardware._role_map:
            resource_name = hardware.resource_by_role(role).name
            if resource_name in prediction.t_per_resource:
                resource_time_by_role_ns[role] = (
                    prediction.t_per_resource[resource_name] * 1e9)
    memory = []
    for path in diagnostics.memory:
        memory.append({
            "path_name": path.path_name,
            "access_kind": path.access_kind,
            "rate_unit": path.rate_unit,
            "initial_transactions": path.initial_transactions,
            "lambda_initial_per_s": path.lambda_initial_per_s,
            "lambda_capacity_per_s": path.lambda_capacity_per_s,
            "workload_slots_per_unit": path.workload_slots_per_unit,
            "hardware_source_slots_per_unit": (
                path.hardware_source_slots_per_unit),
            "source_slots_per_unit": path.source_slots_per_unit,
            "source_lifetime_s": path.source_lifetime_s,
            "source_rate_capacity_per_s": path.source_rate_capacity_per_s,
            "frontend_rate_capacity_per_s": path.frontend_rate_capacity_per_s,
            "steady_time_ns": path.steady_time_s * 1e9,
            "first_request_latency_ns": (
                None if path.first_request_latency_s is None
                else path.first_request_latency_s * 1e9),
            "bottleneck": path.bottleneck,
            "capacity_bound": path.capacity_bound,
            "nodes": [
                {
                    "resource_role": node.resource_role,
                    "transactions": node.transactions,
                    "bytes": node.bytes,
                    "byte_capacity_per_s": node.byte_capacity_per_s,
                    "request_capacity_per_s": node.request_capacity_per_s,
                    "service_capacity_per_s": node.service_capacity_per_s,
                    "arrival_rate_per_s": node.arrival_rate_per_s,
                    "rho": node.rho,
                    "queue_wait_ns": (
                        None if node.queue_wait_s is None
                        else node.queue_wait_s * 1e9),
                    "capacity_time_ns": node.capacity_time_s * 1e9,
                    "saturated": node.saturated,
                }
                for node in path.nodes
            ],
            "queues": [_json_value(queue) for queue in path.queues],
        })
    pipeline = None
    if diagnostics.pipeline is not None:
        pipeline = {
            "execution_time_ns": diagnostics.pipeline.execution_time_s * 1e9,
            "initiation_interval_ns": (
                diagnostics.pipeline.initiation_interval_s * 1e9),
            "worker_makespan_ns": {
                name: value * 1e9
                for name, value in diagnostics.pipeline.worker_makespan_s.items()
            },
            "stage_service_ns": {
                name: value * 1e9
                for name, value in diagnostics.pipeline.stage_service_s.items()
            },
            "critical_path": list(diagnostics.pipeline.critical_path),
            "encoded_task_diagnostic_count": len(diagnostics.pipeline.tasks),
        }
    return {
        "total_ns": prediction.t_total_ns,
        "launch_dispatch_ns": prediction.t_launch_s * 1e9,
        "bottleneck_group": prediction.bottleneck_group,
        "time_per_group_ns": {
            name: value * 1e9 for name, value in prediction.t_per_group.items()
        },
        "time_per_resource_ns": {
            name: value * 1e9
            for name, value in prediction.t_per_resource.items()
        },
        "resource_time_by_role_ns": resource_time_by_role_ns,
        "compute_time_by_role_ns": {
            name: value * 1e9
            for name, value in diagnostics.compute_time_s.items()
        },
        "memory": memory,
        "memory_access_classes": [
            _json_value(value) for value in diagnostics.memory_access_classes
        ],
        "memory_arbitration": [
            _json_value(value) for value in diagnostics.memory_arbitration
        ],
        "pipeline": pipeline,
        "critical_path": list(diagnostics.critical_path),
        "support_level": diagnostics.support_level,
        "limitations": list(diagnostics.limitations),
    }


def _nvidia_l2_directional_summary(
        sections: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return native NCU L2 source traffic without merging directions."""

    l2_table = sections["Memory Workload Analysis Tables"]["L2 Cache"]
    sector_fields = {
        "l2_source_read_bytes": (
            "lts__t_sectors_srcunit_tex_op_read.sum [sector]"),
        "l2_source_write_bytes": (
            "lts__t_sectors_srcunit_tex_op_write.sum [sector]"),
        "l2_read_lookup_hit_bytes": (
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum [sector]"),
        "l2_read_lookup_miss_bytes": (
            "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum [sector]"),
        "l2_write_lookup_hit_bytes": (
            "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum [sector]"),
        "l2_write_lookup_miss_bytes": (
            "lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum [sector]"),
        "l2_source_atomic_alu_bytes": (
            "lts__t_sectors_srcunit_tex_op_atom_dot_alu.sum [sector]"),
        "l2_source_atomic_cas_bytes": (
            "lts__t_sectors_srcunit_tex_op_atom_dot_cas.sum [sector]"),
        "l2_source_reduction_bytes": (
            "lts__t_sectors_srcunit_tex_op_red.sum [sector]"),
    }
    values = {}
    invalid_fields = []
    for name, field in sector_fields.items():
        sectors = to_float(l2_table[field])
        if not math.isfinite(sectors):
            raise ValueError(
                f"NCU L2 sector count {field!r} must be finite: "
                f"{sectors!r}")
        if sectors < 0:
            # This profiler field is a raw observation, not a physical model
            # input.  A negative sector count cannot establish a directional
            # traffic contract, but it must not prevent the report from
            # retaining the remaining valid evidence.  Preserve the missing
            # comparison explicitly instead of turning the value into zero.
            invalid_fields.append(field)
            values[name] = None
        else:
            values[name] = sectors * NVIDIA_L2_SECTOR_BYTES
    for direction in ("read", "write"):
        total = values[f"l2_source_{direction}_bytes"]
        hit = values[f"l2_{direction}_lookup_hit_bytes"]
        miss = values[f"l2_{direction}_lookup_miss_bytes"]
        classified = None if hit is None or miss is None else hit + miss
        # NCU may collect these metrics in different replay passes.  Preserve
        # both native values and record whether they close; never force one to
        # be reconstructed from the other.
        values[f"l2_{direction}_lookup_partition_closes"] = (
            total is not None
            and classified is not None
            and math.isclose(total, classified, rel_tol=0.0, abs_tol=0.0))
    values["l2_directional_read_write_comparable"] = (
        not invalid_fields
        and values["l2_source_atomic_alu_bytes"] == 0.0
        and values["l2_source_atomic_cas_bytes"] == 0.0
        and values["l2_source_reduction_bytes"] == 0.0)
    values["l2_directional_counter_status"] = (
        "complete" if not invalid_fields else "invalid_raw_counter")
    values["l2_directional_invalid_fields"] = invalid_fields
    return values


def _optional_ascend_kb_bytes(
        raw: Mapping[str, Any], field: str) -> float | None:
    if field not in raw or raw[field] == "NA":
        return None
    kilobytes = to_float(raw[field])
    if not math.isfinite(kilobytes) or kilobytes < 0:
        raise ValueError(
            f"Ascend counter {field!r} must be finite and non-negative: "
            f"{kilobytes!r}")
    return kilobytes * 1024.0


def _optional_ascend_percent(
        raw: Mapping[str, Any], field: str) -> float | None:
    if field not in raw or raw[field] in ("", "NA", "N/A"):
        return None
    percentage = to_float(raw[field])
    if (not math.isfinite(percentage)
            or percentage < 0.0 or percentage > 100.0):
        raise ValueError(
            f"Ascend counter {field!r} must be within [0, 100]: "
            f"{percentage!r}")
    return percentage


def _observation_summary(profiler_record, profiler_metrics) -> Mapping[str, Any]:
    sections = profiler_record["Sections"]
    memory_chart = sections["Memory Workload Analysis Chart"]["Memory Chart"]
    instruction_count = None
    if "Instruction Statistics" in sections:
        instruction_header = sections["Instruction Statistics"]["Header"]
        instruction_count = to_float(
            instruction_header["Executed Instructions [inst]"])
    native = None
    pipe_mte_pct = None
    pipe_ste_pct = None
    l2_source_interface_bytes = profiler_metrics["l2_total"]
    downstream_global_bytes = profiler_metrics["dram_bytes"]
    canonical_l2_service_bytes = None
    canonical_l2_service_status = None
    if "MetaX" in profiler_record:
        metax = profiler_record["MetaX"]
        scalars = metax["all_scalars"]
        roofline = metax["roofline"]
        pipe_mte_pct = to_float(scalars["AP MTE Duty ratio"])
        pipe_ste_pct = to_float(scalars["AP STE Duty ratio"])
        if "l2c_memacs" in roofline:
            l2c_memory_access_bytes = to_float(roofline["l2c_memacs"])
        else:
            l2c_memory_access_bytes = None
        if "all_memacs" in roofline:
            global_memory_access_bytes = to_float(roofline["all_memacs"])
        else:
            global_memory_access_bytes = None
        for name, value in (
                ("l2c_memacs", l2c_memory_access_bytes),
                ("all_memacs", global_memory_access_bytes)):
            if value is not None and value < 0:
                raise ValueError(
                    f"MetaX {name} cannot be negative: {value}")
        l2_source_interface_bytes = l2c_memory_access_bytes
        downstream_global_bytes = global_memory_access_bytes
        if (l2c_memory_access_bytes is not None
                and global_memory_access_bytes is not None):
            canonical_l2_service_bytes = max(
                l2c_memory_access_bytes, global_memory_access_bytes)
            canonical_l2_service_status = (
                "max_metax_l2c_source_and_downstream_global")
        else:
            canonical_l2_service_status = (
                "evidence_pending_missing_metax_traffic_counter")
        if "core_clk" in roofline:
            core_clock_ghz = roofline["core_clk"]
        else:
            core_clock_ghz = None
        native = {
            "backend": "metax_mcprofiler",
            "source": metax["source"],
            "duration_source": metax["dur_source"],
            "instrumented_during_ns": metax["during"],
            "core_clock_ghz": core_clock_ghz,
            "workgroups": scalars["WORKGROUPS"],
            "waves": scalars["WAVES"],
            "total_instructions": scalars["Total Instructions"],
            "average_wave_life_cycles": scalars[
                "Average Wave life cycles"],
            "dnoc_read_average_latency_cycles": scalars[
                "Dnoc Read Average Latency"],
            "vls_pipeline_stall_cycles": metax["charts"][
                "ISU stall cycles layout"]["vls_pipeline_stall"],
            "vls_wdata_stall_cycles": metax["charts"][
                "ISU stall cycles layout"]["vls_wdata_stall"],
            "mma_duty_pct": to_float(scalars["AP MMA Duty ratio"]),
            "mte_duty_pct": pipe_mte_pct,
            "ste_duty_pct": pipe_ste_pct,
            "compute_instructions": scalars["Compute Instructions"],
            "memory_instructions": scalars["Memory Instructions"],
            "global_read_instructions": scalars["Global Read Instructions"],
            "global_write_instructions": scalars["Global Write Instructions"],
            "constant_read_instructions": scalars[
                "Constant Read Instructions"],
            "constant_read_sl1_instructions": scalars[
                "Constant Read SL1 Instructions"],
            "constant_read_l2_instructions": scalars[
                "Constant Read L2 Instructions"],
            "constant_sl1_hit_rate_pct": scalars["Constant SL1 Hit Rate"],
            "vl1_read_instructions": scalars["VL1 Read Instructions"],
            "vl1_write_instructions": scalars["VL1 Write Instructions"],
            "global_read_bytes": scalars["Global Memory Read bytes"],
            "global_write_bytes": scalars["Global Memory Write bytes"],
            "dnoc_read_requests": scalars["Dnoc Read Req"],
            "dnoc_write_requests": scalars["Dnoc Write Req"],
            "l2c_read_instructions": scalars["L2C Read Instructions"],
            "l2c_write_instructions": scalars["L2C Write Instructions"],
            "l2c_hit_rate_pct": scalars["L2C Hit Rate"],
            "vl1_hit_rate_pct": scalars["VL1 Hit Rate"],
            "l2c_memory_access_bytes": l2c_memory_access_bytes,
            "global_memory_access_bytes": global_memory_access_bytes,
            "canonical_l2_service_bytes": canonical_l2_service_bytes,
        }
    elif "Ascend" in profiler_record:
        ascend = profiler_record["Ascend"]
        raw = ascend["raw"]
        if raw == {}:
            read_main_memory_kb = None
            write_main_memory_kb = None
            gm_to_l1_kb = None
            gm_to_ub_kb = None
            gm_to_local_kb = None
            ub_to_gm_kb = None
            l2_read_hit_rate_pct = None
            l2_write_hit_rate_pct = None
        else:
            read_main_memory_kb = raw["read_main_memory_datas(KB)"]
            write_main_memory_kb = raw["write_main_memory_datas(KB)"]
            if "GM_to_L1_datas(KB)" in raw:
                gm_to_l1_kb = raw["GM_to_L1_datas(KB)"]
            else:
                gm_to_l1_kb = None
            if "GM_to_UB_datas(KB)" in raw:
                gm_to_ub_kb = raw["GM_to_UB_datas(KB)"]
            else:
                gm_to_ub_kb = None
            ub_to_gm_kb = raw["UB_to_GM_datas(KB)"]
            if ascend["op_type"] in ("cube", "mix"):
                gm_to_local_kb = gm_to_l1_kb
                l2_read_hit_rate_pct = _optional_ascend_percent(
                    raw, "aic_read_hit_rate(%)")
                l2_write_hit_rate_pct = _optional_ascend_percent(
                    raw, "aic_write_hit_rate(%)")
            elif ascend["op_type"] == "vector":
                gm_to_local_kb = gm_to_ub_kb
                l2_read_hit_rate_pct = _optional_ascend_percent(
                    raw, "aiv_read_hit_rate(%)")
                l2_write_hit_rate_pct = _optional_ascend_percent(
                    raw, "aiv_write_hit_rate(%)")
            else:
                raise ValueError(
                    "unknown Ascend execution domain in mechanism report: "
                    f"{ascend['op_type']!r}")
        read_main_memory_bytes = _optional_ascend_kb_bytes(
            raw, "read_main_memory_datas(KB)")
        write_main_memory_bytes = _optional_ascend_kb_bytes(
            raw, "write_main_memory_datas(KB)")
        if (read_main_memory_bytes is None
                or l2_read_hit_rate_pct is None):
            derived_dram_read_bytes = None
        else:
            derived_dram_read_bytes = (
                read_main_memory_bytes
                * (1.0 - l2_read_hit_rate_pct / 100.0))
        canonical_l2_service_status = (
            "not_comparable_as_one_aggregate_for_ascend_mte2_mte3")
        native = {
            "backend": "ascend_msprof",
            "op_type": ascend["op_type"],
            "block_dim": ascend["block_dim"],
            "mix_block_dim": ascend["mix_block_dim"],
            "cube_fops": ascend["cube_fops"],
            "legacy_dram_read_bytes": ascend["dram_read_bytes"],
            "dram_read_bytes": derived_dram_read_bytes,
            "gm_to_l1_bytes": ascend["gm_to_l1_bytes"],
            "l2_hit_rate": ascend["l2_hit_rate"],
            "mte2_tile_count": ascend["mte2_tile_count"],
            "read_main_memory_kb": read_main_memory_kb,
            "write_main_memory_kb": write_main_memory_kb,
            "read_main_memory_bytes": read_main_memory_bytes,
            "write_main_memory_bytes": write_main_memory_bytes,
            "gm_to_l1_kb": gm_to_l1_kb,
            "gm_to_ub_kb": gm_to_ub_kb,
            "gm_to_local_kb": gm_to_local_kb,
            "ub_to_gm_kb": ub_to_gm_kb,
            "raw_gm_to_l1_bytes": _optional_ascend_kb_bytes(
                raw, "GM_to_L1_datas(KB)"),
            "gm_to_ub_bytes": _optional_ascend_kb_bytes(
                raw, "GM_to_UB_datas(KB)"),
            "gm_to_local_bytes": (
                None if gm_to_local_kb is None
                else to_float(gm_to_local_kb) * 1024.0),
            "ub_to_gm_bytes": _optional_ascend_kb_bytes(
                raw, "UB_to_GM_datas(KB)"),
            "l0c_to_gm_bytes": _optional_ascend_kb_bytes(
                raw, "L0C_to_GM_datas(KB)"),
            "l2_read_hit_rate_pct": l2_read_hit_rate_pct,
            "l2_write_hit_rate_pct": l2_write_hit_rate_pct,
            "aic_cube_time_ns": ascend["aic_cube_time_ns"],
            "aic_mte2_time_ns": ascend["aic_mte2_time_ns"],
            "aic_scalar_time_ns": ascend["aic_scalar_time_ns"],
            "aiv_scalar_time_ns": ascend["aiv_scalar_time_ns"],
            "aiv_vec_time_ns": ascend["aiv_vec_time_ns"],
        }
    else:
        nvidia_l2 = _nvidia_l2_directional_summary(sections)
        native = {
            "backend": "nvidia_ncu",
            "executed_instructions": instruction_count,
            **nvidia_l2,
            "dram_read_bytes": to_float(
                memory_chart["dram__bytes_read.sum [byte]"]),
            "dram_write_bytes": to_float(
                memory_chart["dram__bytes_write.sum [byte]"]),
        }
        canonical_l2_service_bytes = max(
            l2_source_interface_bytes, downstream_global_bytes)
        canonical_l2_service_status = (
            "max_ncu_source_interface_and_downstream_dram")
    if canonical_l2_service_status is None:
        raise RuntimeError("observation has no canonical L2 service status")
    return {
        "profiler_header_duration_ns_not_scoring_target": (
            profiler_metrics["duration_ns"]),
        "sm_frequency_hz": profiler_metrics["sm_freq_hz"],
        "dram_frequency_hz": profiler_metrics["dram_freq_hz"],
        "elapsed_cycles": profiler_metrics["elapsed_cycles"],
        "active_cycles": profiler_metrics["active_cycles"],
        "sm_active_fraction": profiler_metrics["sm_active_frac"],
        "pipe_tensor_pct": profiler_metrics["tensor_pct"],
        "pipe_fma_pct": profiler_metrics["fma_pct"],
        "pipe_alu_pct": profiler_metrics["alu_pct"],
        "pipe_sfu_pct": profiler_metrics["sfu_pct"],
        "pipe_lsu_pct": profiler_metrics["lsu_pct"],
        # MetaX MTE is the vector-compute/global-memory engine and STE is the
        # scalar/control engine.  They are not NVIDIA FMA/ALU pipe aliases;
        # preserve their native names for cross-hardware reporting.
        "pipe_mte_pct": pipe_mte_pct,
        "pipe_ste_pct": pipe_ste_pct,
        "l2_interface_read_bytes": to_float(
            memory_chart["l1tex__m_xbar2l1tex_read_bytes.sum [byte]"]),
        "l2_interface_write_bytes": to_float(
            memory_chart["l1tex__m_l1tex2xbar_write_bytes.sum [byte]"]),
        "l2_interface_total_bytes": profiler_metrics["l2_total"],
        "l2_source_interface_bytes": l2_source_interface_bytes,
        "downstream_global_bytes": downstream_global_bytes,
        "canonical_l2_service_bytes": canonical_l2_service_bytes,
        "canonical_l2_service_status": canonical_l2_service_status,
        "dram_read_bytes": to_float(
            memory_chart["dram__bytes_read.sum [byte]"]),
        "dram_write_bytes": to_float(
            memory_chart["dram__bytes_write.sum [byte]"]),
        "dram_total_bytes": profiler_metrics["dram_bytes"],
        "l1_hit_rate": profiler_metrics["l1_hit_rate"],
        "l2_hit_rate": profiler_metrics["l2_hit_rate"],
        "ldgsts_bypass_bytes": profiler_metrics["ldgsts_bypass"],
        "grid": [
            profiler_metrics["grid_x"],
            profiler_metrics["grid_y"],
            profiler_metrics["grid_z"],
        ],
        "grid_size": profiler_metrics["grid_size"],
        "block_size": profiler_metrics["block_size"],
        "registers_per_thread": profiler_metrics["regs_per_thread"],
        "shared_memory_per_block_bytes": profiler_metrics["smem_per_block"],
        "achieved_occupancy_pct": profiler_metrics["achieved_occupancy_pct"],
        "theoretical_occupancy_pct": (
            profiler_metrics["theoretical_occupancy_pct"]),
        "block_limit_registers": profiler_metrics["block_limit_registers"],
        "native": native,
    }


def _oracle_summary(
        oracle_observation, oracle_unavailable_reason) -> Mapping[str, Any]:
    if oracle_observation is None:
        if (not isinstance(oracle_unavailable_reason, str)
                or not oracle_unavailable_reason):
            raise ValueError(
                "unavailable Oracle observation requires a non-empty reason")
        return {
            "available": False,
            "unavailable_reason": oracle_unavailable_reason,
            "backend": None,
            "coverage": None,
            "target_derived_paths": [],
            "field_provenance": {},
            "unavailable_fields": {},
            "observed_input": None,
        }
    provenance = {
        path: {
            "kind": source.kind,
            "provider": source.provider,
            "metric": source.metric,
            "uses_target_duration": source.uses_target_duration,
        }
        for path, source in oracle_observation.field_provenance.items()
    }
    return {
        "available": True,
        "unavailable_reason": None,
        "backend": oracle_observation.backend,
        "coverage": oracle_observation.coverage,
        "target_derived_paths": list(oracle_observation.target_derived_paths),
        "field_provenance": provenance,
        "unavailable_fields": dict(oracle_observation.unavailable_fields),
        "observed_input": _prediction_input_summary(
            oracle_observation.prediction_input),
    }


def _single_counterfactual_summary(
        kernel_spec, oracle_observation, hardware) -> list[Mapping[str, Any]]:
    """Evaluate every profiler-backed formula input one at a time."""

    if oracle_observation is None:
        return []
    predictor = create_predictor(kernel_spec.kind)
    rows = []
    for candidate in single_counterfactual_inputs(oracle_observation):
        prediction = predictor.oracle(
            kernel_spec, hardware, candidate.prediction_input)
        rows.append({
            "component": candidate.component,
            "input_path": candidate.input_path,
            "unit": candidate.unit,
            "provider": candidate.provenance.provider,
            "metric": candidate.provenance.metric,
            "provenance_kind": candidate.provenance.kind,
            "uses_target_duration": candidate.provenance.uses_target_duration,
            "predicted_ns": prediction.t_total_ns,
            "bottleneck_group": prediction.bottleneck_group,
        })
    return rows


def capture_kernel_mechanisms(
        *, kernel_spec, profiler_record, profiler_metrics, hardware,
        forward_prediction, oracle_observation, oracle_unavailable_reason,
        timing_target_ns,
        timing_target_source, timing_target_unavailable_reason,
        kernel_ordinal) -> Mapping[str, Any]:
    """Capture existing model inputs/outputs without recomputing a formula."""

    analytical_input, compulsory_bytes = create_predictor(
        kernel_spec.kind).forward_resource_input(kernel_spec, hardware)
    # Oracle deliberately owns a separately normalized input.  For CANN it
    # may also retain an expanded source DAG while Forward uses the equivalent
    # compact worker envelope.  Capture both representations independently;
    # requiring structural equality would reject valid Forward diagnostics.
    if timing_target_ns is None:
        if timing_target_source is not None:
            raise ValueError(
                "unobserved kernel timing cannot have a timing source")
        if (not isinstance(timing_target_unavailable_reason, str)
                or not timing_target_unavailable_reason.strip()):
            raise ValueError(
                "unobserved kernel timing requires an unavailable reason")
        timing_target = {
            "availability": "unobserved",
            "duration_ns": None,
            "source": None,
            "unavailable_reason": timing_target_unavailable_reason,
        }
    else:
        if (not isinstance(timing_target_ns, (int, float))
                or isinstance(timing_target_ns, bool)
                or not math.isfinite(float(timing_target_ns))
                or timing_target_ns <= 0):
            raise ValueError(
                "observed kernel timing must be finite and positive")
        if (not isinstance(timing_target_source, str)
                or not timing_target_source.strip()):
            raise ValueError(
                "observed kernel timing requires a timing source")
        if timing_target_unavailable_reason is not None:
            raise ValueError(
                "observed kernel timing cannot have an unavailable reason")
        timing_target = {
            "availability": "observed",
            "duration_ns": timing_target_ns,
            "source": timing_target_source,
            "unavailable_reason": None,
        }
    return {
        "kernel_ordinal": kernel_ordinal,
        "kernel_name": profiler_metrics["kernel_name"],
        "kernel_kind": kernel_spec.kind.value,
        "hardware_id": kernel_spec.hardware_id,
        "shape": dict(kernel_spec.shape),
        "dtype": kernel_spec.dtype,
        "variant": kernel_spec.variant,
        "attributes": _json_value(kernel_spec.attributes),
        "observed_launch": _launch_summary(kernel_spec.launch),
        "timing_target": timing_target,
        "observation": _observation_summary(
            profiler_record, profiler_metrics),
        "analytical_input": _prediction_input_summary(analytical_input),
        "compulsory_bytes": compulsory_bytes,
        "forward": _prediction_summary(forward_prediction, hardware),
        "oracle": _oracle_summary(
            oracle_observation, oracle_unavailable_reason),
        "single_counterfactuals": _single_counterfactual_summary(
            kernel_spec, oracle_observation, hardware),
    }


def _validate_output_directory(path: Path) -> Path:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(
            "mechanism-validation output must be a run directory below "
            f"{root}")
    return resolved


def _hardware_work(hardware_id: str):
    raw_data = load_kernel_dataset(hardware_id, DATASET_FAMILIES)
    hardware_config = get_hardware_config_path(hardware_id)
    packed = [
        (
            task_key,
            entry,
            hardware_config,
            True,
            True,
            False,
            0.0,
            False,
            None,
            True,
        )
        for task_key, entry in sorted(raw_data.items())
    ]
    return hardware_config, packed


def _hardware_document(hardware_id, hardware_config, results):
    tasks = []
    for result in results:
        task_key, metadata, oracle_pair, forward_pair, _, _, kernels = result
        if not isinstance(forward_pair, tuple) or len(forward_pair) != 2:
            raise RuntimeError(
                f"{hardware_id} task {task_key!r} has no Forward result")
        tasks.append({
            "case_id": task_key,
            "family": _family(metadata["op_type"]),
            "metadata": metadata,
            "oracle_pair": oracle_pair,
            "forward_pair": list(forward_pair),
            "kernels": kernels,
        })
    return {
        "hardware_id": hardware_id,
        "hardware_config": str(Path(hardware_config).relative_to(REPOSITORY_ROOT)),
        "tasks": tasks,
    }


def _collect_hardware_matrix(jobs: int) -> list[Mapping[str, Any]]:
    """Collect all hardware with one interleaved process pool."""

    work = {}
    maximum_cases = 0
    for hardware_id, _ in HARDWARE:
        hardware_config, packed = _hardware_work(hardware_id)
        work[hardware_id] = {
            "hardware_config": hardware_config,
            "packed": packed,
            "results": [],
        }
        maximum_cases = max(maximum_cases, len(packed))
    owners, results = interleave_and_process(
        work, [hardware_id for hardware_id, _ in HARDWARE],
        maximum_cases, jobs, _process_op_star)
    for hardware_id, result in zip(owners, results, strict=True):
        work[hardware_id]["results"].append(result)
    return [
        _hardware_document(
            hardware_id,
            work[hardware_id]["hardware_config"],
            work[hardware_id]["results"],
        )
        for hardware_id, _ in HARDWARE
    ]


def interleave_and_process(work, hardware_order, maximum_cases, jobs, process):
    """Run one interleaved process pool over every hardware's packed cases.

    Cases are interleaved round-robin across hardware so each worker sees the
    same mix the serial loop would, and the result list carries the hardware
    that produced each entry.  Serial for one worker or a single case.

    ``process`` is the caller's worker callable rather than a name imported
    here: each module keeps its own module-level binding so a test or caller
    can substitute the worker where it is defined.
    """

    interleaved = []
    owners = []
    for case_index in range(maximum_cases):
        for hardware_id in hardware_order:
            packed = work[hardware_id]["packed"]
            if case_index < len(packed):
                interleaved.append(packed[case_index])
                owners.append(hardware_id)
    if jobs == 1 or len(interleaved) <= 1:
        results = [process(item) for item in interleaved]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(process, interleaved))
    return owners, results


def _metric_summary(pairs: Sequence[tuple[float, float]]) -> Mapping[str, Any]:
    if not pairs:
        raise ValueError("metric summary requires at least one pair")
    predicted = np.asarray([pair[0] for pair in pairs], dtype=float)
    observed = np.asarray([pair[1] for pair in pairs], dtype=float)
    if (not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(observed))):
        raise ValueError("latency metrics require finite values")
    if np.any(predicted <= 0) or np.any(observed <= 0):
        raise ValueError("latency metrics require positive values")
    signed = (predicted - observed) / observed * 100.0
    absolute = np.abs(signed)
    return {
        "case_count": len(pairs),
        "mape_pct": float(np.mean(absolute)),
        "mdape_pct": float(np.percentile(absolute, 50)),
        "p50_ape_pct": float(np.percentile(absolute, 50)),
        "p90_ape_pct": float(np.percentile(absolute, 90)),
        "p99_ape_pct": float(np.percentile(absolute, 99)),
        "max_ape_pct": float(np.max(absolute)),
        "weighted_ape_pct": float(
            np.sum(np.abs(predicted - observed)) / np.sum(observed) * 100.0),
        "mean_signed_error_pct": float(np.mean(signed)),
        "median_signed_error_pct": float(np.percentile(signed, 50)),
    }


def _format_percentage(value: float | None, *, signed: bool = False) -> str:
    """Format an optional percentage without turning missing evidence into zero."""

    if value is None:
        return "n/a"
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("percentage display value must be finite")
    if signed:
        return f"{numeric:+.1f}%"
    return f"{numeric:.1f}%"


def _result_scope_title(result_scope: str) -> str:
    if result_scope == "real_kernel":
        return "Real kernel invocation"
    elif result_scope == "aggregate_task":
        return "Complete task"
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")


def _comparison_status(row: Mapping[str, Any]) -> str:
    status = row["comparison_status"]
    if status not in COMPARISON_STATUSES:
        raise ValueError(f"unknown accuracy comparison status: {status!r}")
    reason = row["comparison_reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("accuracy comparison reason must be non-empty")
    return status


def _observed_duration_ns(
        row: Mapping[str, Any], result_scope: str) -> float:
    if result_scope == "real_kernel":
        return float(row["actual_ns"])
    elif result_scope == "aggregate_task":
        return float(row["observed_ns"])
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")


def _kernel_ordinal_for_scope(
        row: Mapping[str, Any], result_scope: str) -> int | None:
    if result_scope == "real_kernel":
        ordinal = row["kernel_ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("real-kernel ordinal must be an integer")
        if ordinal < 0:
            raise ValueError("real-kernel ordinal must be non-negative")
        return ordinal
    elif result_scope == "aggregate_task":
        return None
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")


def _kernel_ordinal_sort_value(
        row: Mapping[str, Any], result_scope: str) -> int:
    ordinal = _kernel_ordinal_for_scope(row, result_scope)
    if ordinal is None:
        return -1
    return ordinal


def _kernel_ordinal_suffix(
        row: Mapping[str, Any], result_scope: str) -> str:
    ordinal = _kernel_ordinal_for_scope(row, result_scope)
    if ordinal is None:
        return ""
    return f" k{ordinal}"


def _plot_case_id(case_id: str, width: int = 34) -> str:
    """Wrap a case identifier at underscores without changing its text."""

    if not isinstance(case_id, str) or not case_id:
        raise ValueError("plot case identifier must be a non-empty string")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("plot case identifier width must be positive")
    tokens = case_id.split("_")
    lines = []
    current = tokens[0]
    for token in tokens[1:]:
        candidate = f"{current}_{token}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = token
    lines.append(current)
    return "\n".join(lines)


def _support_level(kernels: Sequence[Mapping[str, Any]]) -> str:
    ranks = {level: rank for rank, level in enumerate(SUPPORT_LEVELS)}
    levels = [
        kernel["mechanisms"]["forward"]["support_level"]
        for kernel in kernels
    ]
    unknown = set(levels) - set(ranks)
    if unknown:
        raise ValueError(f"unknown Forward support levels: {sorted(unknown)}")
    return max(levels, key=lambda value: ranks[value])


def _kernel_launch_class(mechanism: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("hardware_id", "kernel_kind", "variant", "dtype"):
        value = mechanism[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"physical bucket requires non-empty {name}: {value!r}")
    launch = mechanism["observed_launch"]
    if not isinstance(launch, Mapping):
        raise TypeError("physical bucket launch must be a mapping")
    grid = launch["grid"]
    block = launch["block"]
    if (not isinstance(grid, list) or len(grid) != 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in grid)):
        raise ValueError("physical bucket requires a positive 3-D launch grid")
    if (not isinstance(block, list) or len(block) != 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in block)):
        raise ValueError("physical bucket requires a positive 3-D launch block")
    for name in ("algorithm", "execution_variant"):
        value = launch[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"physical bucket requires non-empty launch {name}: "
                f"{value!r}")
    cluster = launch["cluster"]
    if (cluster is not None
            and (not isinstance(cluster, list) or len(cluster) != 3
                 or any(isinstance(value, bool)
                        or not isinstance(value, int) or value <= 0
                        for value in cluster))):
        raise ValueError(
            "physical bucket launch cluster must be null or a positive 3-D "
            "list")
    source_program_type = launch["source_program_type"]
    if (source_program_type is not None
            and (not isinstance(source_program_type, str)
                 or not source_program_type.strip())):
        raise ValueError(
            "physical bucket source_program_type must be null or a non-empty "
            "string")
    tiling = launch["tiling"]
    if tiling is not None:
        if not isinstance(tiling, Mapping):
            raise TypeError("physical bucket tiling must be a mapping or null")
        expected_tiling_fields = {
            "tile_shape", "pipeline_depth", "split_k", "num_kv_splits",
        }
        if set(tiling) != expected_tiling_fields:
            raise ValueError(
                "physical bucket tiling fields must be exactly "
                f"{sorted(expected_tiling_fields)}")
        tile_shape = tiling["tile_shape"]
        if (not isinstance(tile_shape, Mapping)
                or any(not isinstance(name, str) or not name
                       for name in tile_shape)
                or any(isinstance(value, bool)
                       or not isinstance(value, int) or value <= 0
                       for value in tile_shape.values())):
            raise ValueError(
                "physical bucket tile_shape must be an explicit mapping of "
                "positive integer dimensions; an empty mapping denotes a "
                "source launch with no tile-shape fields")
        for name in ("pipeline_depth", "split_k", "num_kv_splits"):
            value = tiling[name]
            if (value is not None
                    and (isinstance(value, bool)
                         or not isinstance(value, int) or value <= 0)):
                raise ValueError(
                    f"physical bucket tiling {name} must be null or a "
                    f"positive integer: {value!r}")
    parameters = launch["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("physical bucket launch parameters must be a mapping")
    controls = {}
    for name in PHYSICAL_BUCKET_DISCRETE_CONTROLS:
        if name in parameters:
            controls[name] = parameters[name]
        else:
            controls[name] = None
    for name, value in controls.items():
        if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not float(value).is_integer()):
            raise ValueError(
                f"physical bucket control {name} must be a finite integer "
                f"or null: {value!r}")
    return {
        "hardware_id": mechanism["hardware_id"],
        "kernel_kind": mechanism["kernel_kind"],
        "variant": mechanism["variant"],
        "dtype": mechanism["dtype"],
        "execution_variant": launch["execution_variant"],
        "algorithm": launch["algorithm"],
        "grid_rank": max(
            1, sum(dimension > 1 for dimension in grid)),
        "block": block,
        "cluster": cluster,
        "source_program_type": source_program_type,
        "tiling": tiling,
        "discrete_controls": controls,
    }


def _task_execution_signature(task: Mapping[str, Any]) -> str:
    signature = []
    for kernel in task["kernels"]:
        signature.append(_kernel_launch_class(kernel["mechanisms"]))
    if not signature:
        raise ValueError("task execution signature requires at least one kernel")
    return json.dumps(signature, separators=(",", ":"), sort_keys=True)


def _physical_bucket(
        *, hardware_id: str, family: str, metadata: Mapping[str, Any],
        result_scope: str, execution_signature: str) -> Mapping[str, Any]:
    if result_scope not in ("real_kernel", "aggregate_task"):
        raise ValueError(f"unknown result scope: {result_scope!r}")
    dimensions = {
        "schema": BUCKET_SCHEMA_VERSION,
        "result_scope": result_scope,
        "hardware_id": hardware_id,
        "execution_signature": execution_signature,
    }
    if family == "FlashAttention":
        layout = metadata["attention_layout"]
        if not isinstance(layout, str) or not layout:
            raise ValueError("FlashAttention bucket requires a layout")
        dimensions["layout"] = layout
    elif family in ("GEMM", "FusedMoE", "Vector"):
        pass
    else:
        raise ValueError(f"unsupported physical-bucket family: {family!r}")
    if result_scope == "real_kernel":
        bucket_basis = "static_kernel_launch_class"
    elif result_scope == "aggregate_task":
        bucket_basis = "ordered_static_launch_graph"
    else:
        raise AssertionError(f"unhandled result scope: {result_scope!r}")
    bucket_dimensions = json.dumps(
        dimensions, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(bucket_dimensions.encode("utf-8")).hexdigest()
    bucket_id = f"{BUCKET_SCHEMA_VERSION}:{result_scope}:{digest}"
    return {
        "bucket_schema": BUCKET_SCHEMA_VERSION,
        "physical_bucket_id": bucket_id,
        "physical_bucket_short_id": digest[:12],
        "bucket_dimensions": bucket_dimensions,
        "bucket_basis": bucket_basis,
    }


def _task_axis_summary(
        metadata: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    result = {
        "gemm_m": None,
        "gemm_n": None,
        "gemm_k": None,
        "query_length": None,
        "kv_length": None,
        "executed_attention_splits": None,
        "attention_layout": None,
        "moe_experts": None,
        "moe_topk": None,
        "moe_max_tokens_per_expert": None,
        "vector_tokens": None,
    }
    if family == "GEMM":
        result["gemm_m"] = metadata["gemm_m"]
        result["gemm_n"] = metadata["gemm_n"]
        result["gemm_k"] = metadata["gemm_k"]
    elif family == "FlashAttention":
        result["query_length"] = metadata["query_length"]
        result["kv_length"] = metadata["kv_length"]
        result["executed_attention_splits"] = (
            metadata["executed_attention_splits"])
        result["attention_layout"] = metadata["attention_layout"]
    elif family == "FusedMoE":
        result["moe_experts"] = metadata["moe_experts"]
        result["moe_topk"] = metadata["moe_topk"]
        result["moe_max_tokens_per_expert"] = (
            metadata["moe_max_tokens_per_expert"])
    elif family == "Vector":
        result["vector_tokens"] = metadata["vector_tokens"]
    else:
        raise ValueError(f"unsupported report family: {family!r}")
    return result


def _task_rows(documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        comparison_evidence = _canonical_capture_comparison_evidence(
            hardware_id)
        for task in document["tasks"]:
            predicted_ns, observed_ns = task["forward_pair"]
            family = task["family"]
            metadata = task["metadata"]
            variant_signature = "/".join(
                kernel["mechanisms"]["variant"]
                for kernel in task["kernels"]
            )
            execution_signature = _task_execution_signature(task)
            bucket = _physical_bucket(
                hardware_id=hardware_id,
                family=family,
                metadata=metadata,
                result_scope="aggregate_task",
                execution_signature=execution_signature,
            )
            rows.append({
                "result_scope": "aggregate_task",
                "hardware_id": hardware_id,
                "family": family,
                "case_id": task["case_id"],
                "predicted_ns": predicted_ns,
                "observed_ns": observed_ns,
                "timing_source": metadata["actual_source"],
                "signed_error_pct": (
                    (predicted_ns - observed_ns) / observed_ns * 100.0),
                "absolute_error_pct": (
                    abs(predicted_ns - observed_ns) / observed_ns * 100.0),
                "absolute_duration_error_ns": abs(
                    predicted_ns - observed_ns),
                **comparison_evidence,
                "support_level": _support_level(task["kernels"]),
                "kernel_count": len(task["kernels"]),
                "aggregation_contract": TASK_AGGREGATION_CONTRACT,
                "observed_timing_granularity": metadata[
                    "actual_granularity"],
                "variant_signature": variant_signature,
                "task_execution_signature": execution_signature,
                "scale": metadata["eqm"],
                "max_derived_launch_rounds": metadata["max_launch_waves"],
                **_task_axis_summary(metadata, family),
                **bucket,
            })
    return rows


def _error_direction(predicted_ns: float, observed_ns: float) -> str:
    if predicted_ns < observed_ns:
        return "under"
    if predicted_ns > observed_ns:
        return "over"
    return "exact"


def _counterfactual_rows(
        documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Aggregate one-counter-at-a-time kernel predictions at task scope."""

    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        comparison_evidence = _canonical_capture_comparison_evidence(
            hardware_id)
        for task in document["tasks"]:
            forward_ns, observed_ns = task["forward_pair"]
            candidate_by_kernel = []
            components = set()
            for kernel in task["kernels"]:
                entries = kernel["mechanisms"]["single_counterfactuals"]
                index = {}
                for entry in entries:
                    component = entry["component"]
                    if component in index:
                        raise ValueError(
                            "kernel repeats a single-counterfactual component: "
                            f"{task['case_id']!r} {component!r}")
                    index[component] = entry
                    components.add(component)
                candidate_by_kernel.append(index)
            family = task["family"]
            execution_signature = _task_execution_signature(task)
            bucket = _physical_bucket(
                hardware_id=hardware_id,
                family=family,
                metadata=task["metadata"],
                result_scope="aggregate_task",
                execution_signature=execution_signature,
            )
            for component in sorted(components):
                counterfactual_ns = 0.0
                replacement_count = 0
                input_paths = set()
                providers = set()
                metrics = set()
                provenance_kinds = set()
                units = set()
                for kernel, index in zip(
                        task["kernels"], candidate_by_kernel, strict=True):
                    if component in index:
                        candidate = index[component]
                        if candidate["uses_target_duration"]:
                            raise ValueError(
                                "counterfactual candidate consumes target "
                                f"duration: {component!r}")
                        counterfactual_ns += candidate["predicted_ns"]
                        replacement_count += 1
                        input_paths.add(candidate["input_path"])
                        providers.add(candidate["provider"])
                        metrics.add(candidate["metric"])
                        provenance_kinds.add(candidate["provenance_kind"])
                        units.add(candidate["unit"])
                    else:
                        counterfactual_ns += kernel[
                            "mechanisms"]["forward"]["total_ns"]
                if replacement_count <= 0:
                    raise RuntimeError(
                        "counterfactual component has no replacements")
                if len(input_paths) != 1 or len(units) != 1:
                    raise ValueError(
                        "counterfactual component disagrees on input path or "
                        f"unit: {component!r}")
                forward_absolute_error_ns = abs(forward_ns - observed_ns)
                counterfactual_absolute_error_ns = abs(
                    counterfactual_ns - observed_ns)
                rows.append({
                    "hardware_id": hardware_id,
                    "family": family,
                    "case_id": task["case_id"],
                    "component": component,
                    "input_path": next(iter(input_paths)),
                    "unit": next(iter(units)),
                    "provider": json.dumps(sorted(providers)),
                    "metric": json.dumps(sorted(metrics)),
                    "provenance_kind": json.dumps(
                        sorted(provenance_kinds)),
                    "kernel_count": len(task["kernels"]),
                    "replaced_kernel_count": replacement_count,
                    "full_kernel_coverage": (
                        replacement_count == len(task["kernels"])),
                    "forward_predicted_ns": forward_ns,
                    "counterfactual_predicted_ns": counterfactual_ns,
                    "observed_ns": observed_ns,
                    "forward_absolute_error_ns": forward_absolute_error_ns,
                    "counterfactual_absolute_error_ns": (
                        counterfactual_absolute_error_ns),
                    "oracle_gain_ns": (
                        forward_absolute_error_ns
                        - counterfactual_absolute_error_ns),
                    "forward_error_direction": _error_direction(
                        forward_ns, observed_ns),
                    "counterfactual_error_direction": _error_direction(
                        counterfactual_ns, observed_ns),
                    **comparison_evidence,
                    **bucket,
                })
    return rows


def _counterfactual_summary_rows(
        counterfactual_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rank buckets by absolute error and counters by cumulative oracle gain."""

    bucket_rows = {}
    for row in task_rows:
        key = (row["hardware_id"], row["physical_bucket_id"])
        if key not in bucket_rows:
            bucket_rows[key] = []
        bucket_rows[key].append(row)
    target_bucket_keys = [
        key for key in bucket_rows
        if key[0] in ("nvidia_a800", "ascend_910b")
    ]
    target_bucket_keys.sort(
        key=lambda key: (
            -sum(float(row["absolute_duration_error_ns"])
                 for row in bucket_rows[key]),
            key,
        ))
    target_bucket_rank = {
        key: rank
        for rank, key in enumerate(target_bucket_keys, start=1)
    }

    grouped = {}
    for row in counterfactual_rows:
        key = (
            row["hardware_id"], row["family"],
            row["physical_bucket_id"], row["component"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    summaries = []
    for key, rows in grouped.items():
        hardware_id, family, bucket_id, component = key
        bucket_key = (hardware_id, bucket_id)
        all_bucket_rows = bucket_rows[bucket_key]
        bucket_observed_ns = math.fsum(
            float(row["observed_ns"]) for row in all_bucket_rows)
        bucket_absolute_error_ns = math.fsum(
            float(row["absolute_duration_error_ns"])
            for row in all_bucket_rows)
        candidate_observed_ns = math.fsum(
            float(row["observed_ns"]) for row in rows)
        forward_absolute_error_ns = math.fsum(
            float(row["forward_absolute_error_ns"]) for row in rows)
        counterfactual_absolute_error_ns = math.fsum(
            float(row["counterfactual_absolute_error_ns"]) for row in rows)
        if bucket_key in target_bucket_rank:
            bucket_rank = target_bucket_rank[bucket_key]
        else:
            bucket_rank = None
        summaries.append({
            "target_bucket_rank": bucket_rank,
            "counter_rank_within_bucket": None,
            "hardware_id": hardware_id,
            "family": family,
            "physical_bucket_id": bucket_id,
            "physical_bucket_short_id": rows[0][
                "physical_bucket_short_id"],
            "bucket_case_count": len(all_bucket_rows),
            "bucket_total_absolute_error_ns": bucket_absolute_error_ns,
            "bucket_wmae_pct": (
                bucket_absolute_error_ns / bucket_observed_ns * 100.0),
            "component": component,
            "input_path": rows[0]["input_path"],
            "unit": rows[0]["unit"],
            "provider": json.dumps(sorted({row["provider"] for row in rows})),
            "metric": json.dumps(sorted({row["metric"] for row in rows})),
            "covered_case_count": len(rows),
            "full_kernel_coverage_case_count": sum(
                bool(row["full_kernel_coverage"]) for row in rows),
            "replaced_kernel_count": sum(
                int(row["replaced_kernel_count"]) for row in rows),
            "forward_absolute_error_ns": forward_absolute_error_ns,
            "counterfactual_absolute_error_ns": (
                counterfactual_absolute_error_ns),
            "cumulative_oracle_gain_ns": (
                forward_absolute_error_ns
                - counterfactual_absolute_error_ns),
            "forward_wmae_pct": (
                forward_absolute_error_ns / candidate_observed_ns * 100.0),
            "counterfactual_wmae_pct": (
                counterfactual_absolute_error_ns
                / candidate_observed_ns * 100.0),
            "underprediction_case_count": sum(
                row["forward_error_direction"] == "under" for row in rows),
            "overprediction_case_count": sum(
                row["forward_error_direction"] == "over" for row in rows),
            "comparison_statuses": json.dumps(sorted({
                row["comparison_status"] for row in rows
            })),
        })
    by_bucket = {}
    for row in summaries:
        key = (row["hardware_id"], row["physical_bucket_id"])
        if key not in by_bucket:
            by_bucket[key] = []
        by_bucket[key].append(row)
    for rows in by_bucket.values():
        rows.sort(key=lambda row: (
            -float(row["cumulative_oracle_gain_ns"]), row["component"]))
        for rank, row in enumerate(rows, start=1):
            row["counter_rank_within_bucket"] = rank
    summaries.sort(key=lambda row: (
        row["target_bucket_rank"] is None,
        float("inf") if row["target_bucket_rank"] is None
        else int(row["target_bucket_rank"]),
        int(row["counter_rank_within_bucket"]),
        row["hardware_id"],
        row["physical_bucket_id"],
        row["component"],
    ))
    return summaries


def _kernel_rows(documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        comparison_evidence = _canonical_capture_comparison_evidence(
            hardware_id)
        for task in document["tasks"]:
            for kernel in task["kernels"]:
                mechanism = kernel["mechanisms"]
                timing_target = mechanism["timing_target"]
                if timing_target["availability"] == "unobserved":
                    if (timing_target["duration_ns"] is not None
                            or timing_target["source"] is not None
                            or not timing_target["unavailable_reason"]):
                        raise ValueError(
                            "unobserved kernel timing target is malformed")
                    continue
                if timing_target["availability"] != "observed":
                    raise ValueError(
                        "unknown kernel timing availability: "
                        f"{timing_target['availability']!r}")
                analytical = mechanism["analytical_input"]
                observation = mechanism["observation"]
                forward = mechanism["forward"]
                launch = mechanism["observed_launch"]
                execution_signature = json.dumps(
                    _kernel_launch_class(mechanism),
                    separators=(",", ":"), sort_keys=True)
                bucket = _physical_bucket(
                    hardware_id=hardware_id,
                    family=task["family"],
                    metadata=task["metadata"],
                    result_scope="real_kernel",
                    execution_signature=execution_signature,
                )
                rows.append({
                    "result_scope": "real_kernel",
                    "hardware_id": hardware_id,
                    "family": task["family"],
                    "op_type": kernel["op_type"],
                    "case_id": task["case_id"],
                    "kernel_ordinal": mechanism["kernel_ordinal"],
                    "kernel_kind": mechanism["kernel_kind"],
                    "variant": mechanism["variant"],
                    "kernel_name": mechanism["kernel_name"],
                    "actual_ns": kernel["actual_ns"],
                    "timing_source": timing_target["source"],
                    "predicted_ns": forward["total_ns"],
                    "signed_error_pct": (
                        (forward["total_ns"] - kernel["actual_ns"])
                        / kernel["actual_ns"] * 100.0),
                    "absolute_error_pct": (
                        abs(forward["total_ns"] - kernel["actual_ns"])
                        / kernel["actual_ns"] * 100.0),
                    "absolute_duration_error_ns": abs(
                        forward["total_ns"] - kernel["actual_ns"]),
                    **comparison_evidence,
                    "support_level": forward["support_level"],
                    "limitations": json.dumps(
                        forward["limitations"], ensure_ascii=False),
                    "bottleneck": forward["bottleneck_group"],
                    "shape": json.dumps(
                        mechanism["shape"], separators=(",", ":"),
                        sort_keys=True),
                    "task_execution_signature": _task_execution_signature(task),
                    "grid_size": analytical["grid_size"],
                    "block_size": analytical["block_size"],
                    "wave_efficiency": analytical["wave_efficiency"],
                    "active_units": analytical["active_units"],
                    "predicted_l2_bytes": analytical[
                        "demand_per_resource"]["l2_cache"],
                    "predicted_dram_bytes": analytical[
                        "demand_per_resource"]["dram"],
                    "observed_l2_source_interface_bytes": observation[
                        "l2_source_interface_bytes"],
                    "observed_downstream_global_bytes": observation[
                        "downstream_global_bytes"],
                    "observed_l2_service_bytes": observation[
                        "canonical_l2_service_bytes"],
                    "observed_l2_service_status": observation[
                        "canonical_l2_service_status"],
                    **_directional_kernel_traffic_fields(mechanism),
                    "l1_hit_rate": observation["l1_hit_rate"],
                    "l2_hit_rate": observation["l2_hit_rate"],
                    "achieved_occupancy_pct": observation[
                        "achieved_occupancy_pct"],
                    "theoretical_occupancy_pct": observation[
                        "theoretical_occupancy_pct"],
                    "pipe_tensor_pct": observation["pipe_tensor_pct"],
                    "pipe_fma_pct": observation["pipe_fma_pct"],
                    "pipe_alu_pct": observation["pipe_alu_pct"],
                    "pipe_sfu_pct": observation["pipe_sfu_pct"],
                    "pipe_mte_pct": observation["pipe_mte_pct"],
                    "pipe_ste_pct": observation["pipe_ste_pct"],
                    "launch_algorithm": launch["algorithm"],
                    "launch_grid_size": launch["grid_size"],
                    "launch_block_size": launch["block_size"],
                    "launch_execution_variant": launch["execution_variant"],
                    "launch_tile_shape": json.dumps(
                        None if launch["tiling"] is None
                        else launch["tiling"]["tile_shape"],
                        sort_keys=True),
                    "launch_num_kv_splits": (
                        None if launch["tiling"] is None
                        else launch["tiling"]["num_kv_splits"]),
                    **_task_axis_summary(task["metadata"], task["family"]),
                    **bucket,
                })
    return rows


def _memory_path_bytes(
        analytical: Mapping[str, Any], *, access_kind: str,
        resource_role: str) -> float:
    """Sum one directional resource interface from the Forward path ledger."""

    if access_kind not in ("read", "write"):
        raise ValueError(
            f"memory-path access kind must be read or write: {access_kind!r}")
    paths = analytical["memory_paths"]
    if not isinstance(paths, list):
        raise TypeError("analytical memory_paths must be a list")
    total = 0.0
    for path in paths:
        path_kind = path["access_kind"]
        if path_kind not in ("read", "write"):
            raise ValueError(
                f"unknown analytical memory-path access kind: {path_kind!r}")
        nodes = path["nodes"]
        if not isinstance(nodes, list):
            raise TypeError("analytical memory-path nodes must be a list")
        if path_kind != access_kind:
            continue
        for node in nodes:
            if node["resource_role"] != resource_role:
                continue
            byte_count = float(node["bytes"])
            if not math.isfinite(byte_count) or byte_count < 0:
                raise ValueError(
                    "analytical memory-path bytes must be finite and "
                    f"non-negative: {byte_count!r}")
            total += byte_count
    return total


def _directional_kernel_traffic_fields(
        mechanism: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expose native directional counters without inventing equivalence."""

    analytical = mechanism["analytical_input"]
    observation = mechanism["observation"]
    native = observation["native"]
    values = {
        "predicted_l2_read_bytes": _memory_path_bytes(
            analytical, access_kind="read", resource_role="l2_cache"),
        "predicted_l2_write_bytes": _memory_path_bytes(
            analytical, access_kind="write", resource_role="l2_cache"),
        "predicted_dram_read_bytes": _memory_path_bytes(
            analytical, access_kind="read", resource_role="dram"),
        "predicted_dram_write_bytes": _memory_path_bytes(
            analytical, access_kind="write", resource_role="dram"),
        "predicted_mte3_write_bytes": _memory_path_bytes(
            analytical, access_kind="write", resource_role="mte3"),
        "observed_xbar_read_bytes": observation["l2_interface_read_bytes"],
        "observed_xbar_write_bytes": observation["l2_interface_write_bytes"],
        "observed_nvidia_l2_source_read_bytes": None,
        "observed_nvidia_l2_source_write_bytes": None,
        "observed_nvidia_l2_read_lookup_hit_bytes": None,
        "observed_nvidia_l2_read_lookup_miss_bytes": None,
        "observed_nvidia_l2_write_lookup_hit_bytes": None,
        "observed_nvidia_l2_write_lookup_miss_bytes": None,
        "observed_nvidia_l2_read_lookup_partition_closes": None,
        "observed_nvidia_l2_write_lookup_partition_closes": None,
        "observed_nvidia_dram_read_in_window_bytes": None,
        "observed_nvidia_dram_writeback_in_window_bytes": None,
        "observed_metax_l2c_source_read_bytes": None,
        "observed_metax_downstream_global_read_bytes": None,
        "observed_metax_downstream_global_write_bytes": None,
        "observed_ascend_gm_to_l1_bytes": None,
        "observed_ascend_gm_to_ub_bytes": None,
        "observed_ascend_ub_to_gm_bytes": None,
        "observed_ascend_l0c_to_gm_bytes": None,
        "observed_ascend_dram_read_derived_bytes": None,
    }
    if native["backend"] == "nvidia_ncu":
        values.update({
            "observed_nvidia_l2_source_read_bytes": native[
                "l2_source_read_bytes"],
            "observed_nvidia_l2_source_write_bytes": native[
                "l2_source_write_bytes"],
            "observed_nvidia_l2_read_lookup_hit_bytes": native[
                "l2_read_lookup_hit_bytes"],
            "observed_nvidia_l2_read_lookup_miss_bytes": native[
                "l2_read_lookup_miss_bytes"],
            "observed_nvidia_l2_write_lookup_hit_bytes": native[
                "l2_write_lookup_hit_bytes"],
            "observed_nvidia_l2_write_lookup_miss_bytes": native[
                "l2_write_lookup_miss_bytes"],
            "observed_nvidia_l2_read_lookup_partition_closes": native[
                "l2_read_lookup_partition_closes"],
            "observed_nvidia_l2_write_lookup_partition_closes": native[
                "l2_write_lookup_partition_closes"],
            "observed_nvidia_dram_read_in_window_bytes": native[
                "dram_read_bytes"],
            "observed_nvidia_dram_writeback_in_window_bytes": native[
                "dram_write_bytes"],
        })
    elif native["backend"] == "metax_mcprofiler":
        values.update({
            "observed_metax_l2c_source_read_bytes": native[
                "l2c_memory_access_bytes"],
            "observed_metax_downstream_global_read_bytes": native[
                "global_read_bytes"],
            "observed_metax_downstream_global_write_bytes": native[
                "global_write_bytes"],
        })
    elif native["backend"] == "ascend_msprof":
        values.update({
            "observed_ascend_gm_to_l1_bytes": native[
                "raw_gm_to_l1_bytes"],
            "observed_ascend_gm_to_ub_bytes": native["gm_to_ub_bytes"],
            "observed_ascend_ub_to_gm_bytes": native["ub_to_gm_bytes"],
            "observed_ascend_l0c_to_gm_bytes": native["l0c_to_gm_bytes"],
            "observed_ascend_dram_read_derived_bytes": native[
                "dram_read_bytes"],
        })
    else:
        raise ValueError(
            f"unknown directional observation backend: {native['backend']!r}")
    return values


def _directional_traffic_row(
        *, task: Mapping[str, Any], mechanism: Mapping[str, Any],
        component: str, resource_role: str, access_kind: str,
        observed: float, provider: str, metric: str,
        comparison_evidence: Mapping[str, str] | None = None,
        ) -> Mapping[str, Any] | None:
    if not math.isfinite(observed) or observed < 0:
        raise ValueError(
            f"directional traffic observation must be finite and non-negative: "
            f"{observed!r}")
    predicted = _memory_path_bytes(
        mechanism["analytical_input"],
        access_kind=access_kind,
        resource_role=resource_role,
    )
    if predicted == 0.0 and observed == 0.0:
        return None
    if comparison_evidence is None:
        evidence = _canonical_capture_comparison_evidence(
            mechanism["hardware_id"])
    else:
        required_evidence_fields = {"comparison_status", "comparison_reason"}
        if set(comparison_evidence) != required_evidence_fields:
            raise ValueError(
                "directional comparison evidence must contain exactly "
                f"{sorted(required_evidence_fields)}")
        evidence = comparison_evidence
    return {
        "hardware_id": mechanism["hardware_id"],
        "family": task["family"],
        "case_id": task["case_id"],
        "kernel_ordinal": mechanism["kernel_ordinal"],
        "variant": mechanism["variant"],
        "kernel_name": mechanism["kernel_name"],
        "actual_ns": mechanism["timing_target"]["duration_ns"],
        "timing_source": mechanism["timing_target"]["source"],
        "component": component,
        "unit": "bytes",
        "predicted": predicted,
        "observed": observed,
        **evidence,
        "provenance_kind": "profiler",
        "provider": provider,
        "metric": metric,
    }


def _nvidia_directional_traffic_rows(
        *, task: Mapping[str, Any],
        mechanism: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Compare A800 read/write traffic without an aggregate path split."""

    native = mechanism["observation"]["native"]
    if native["backend"] != "nvidia_ncu":
        raise ValueError(
            "NVIDIA directional traffic requires an NCU observation")
    observation = mechanism["observation"]
    specifications = [
        (
            "traffic:xbar_read", "l2_cache", "read",
            observation["l2_interface_read_bytes"],
            "l1tex__m_xbar2l1tex_read_bytes.sum",
        ),
        (
            "traffic:xbar_write", "l2_cache", "write",
            observation["l2_interface_write_bytes"],
            "l1tex__m_l1tex2xbar_write_bytes.sum",
        ),
        (
            "traffic:dram_read_in_window", "dram", "read",
            native["dram_read_bytes"], "dram__bytes_read.sum",
        ),
    ]
    if native["l2_directional_read_write_comparable"]:
        # L2 atom/reduction requests are separate native operations.  Only
        # compare the ordinary read/write source counters when those classes
        # are absent from the invocation.
        specifications.extend((
        (
            "traffic:l2_source_read", "l2_cache", "read",
            native["l2_source_read_bytes"],
            "lts__t_sectors_srcunit_tex_op_read.sum * 32 bytes",
        ),
        (
            "traffic:l2_source_write", "l2_cache", "write",
            native["l2_source_write_bytes"],
            "lts__t_sectors_srcunit_tex_op_write.sum * 32 bytes",
        ),
        ))
    rows = []
    for component, role, access_kind, observed, metric in specifications:
        row = _directional_traffic_row(
            task=task,
            mechanism=mechanism,
            component=component,
            resource_role=role,
            access_kind=access_kind,
            observed=float(observed),
            provider="nvidia_ncu",
            metric=metric,
        )
        if row is not None:
            rows.append(row)
    writeback_row = _directional_traffic_row(
        task=task,
        mechanism=mechanism,
        component="traffic:dram_writeback_in_window",
        resource_role="dram",
        access_kind="write",
        observed=float(native["dram_write_bytes"]),
        provider="nvidia_ncu",
        metric="dram__bytes_write.sum",
        comparison_evidence={
            "comparison_status": "not_comparable",
            "comparison_reason": (
                "ncu_kernel_window_writeback_is_not_the_forward_output_store_"
                "obligation_without_independent_eviction_boundary_evidence"),
        },
    )
    if writeback_row is not None:
        rows.append(writeback_row)
    return rows


def _ascend_directional_traffic_rows(
        *, task: Mapping[str, Any],
        mechanism: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Compare pure AIC/AIV transfers at their native directional boundary."""

    native = mechanism["observation"]["native"]
    if native["backend"] != "ascend_msprof":
        raise ValueError(
            "Ascend directional traffic requires an msprof observation")
    if native["op_type"] == "mix":
        # Historical mixed-domain aggregates did not retain the per-core join
        # required to attribute the AIC and AIV directions independently.
        return []
    if native["op_type"] not in ("cube", "vector"):
        raise ValueError(
            "Ascend directional traffic requires cube/vector Op Type, got "
            f"{native['op_type']!r}")
    if native["gm_to_local_bytes"] is None:
        return []
    if native["op_type"] == "cube":
        mte2_component = "traffic:aic_mte2_gm_to_l1"
        mte2_metric = "GM_to_L1_datas"
    elif native["op_type"] == "vector":
        mte2_component = "traffic:aiv_mte2_gm_to_ub"
        mte2_metric = "GM_to_UB_datas"
    else:
        raise AssertionError(
            f"unhandled pure Ascend Op Type: {native['op_type']!r}")
    specifications = [
        (
            mte2_component, "l2_cache", "read",
            native["gm_to_local_bytes"], mte2_metric,
        ),
        (
            "traffic:l2_source_write", "l2_cache", "write",
            native["write_main_memory_bytes"],
            "write_main_memory_datas",
        ),
        (
            "traffic:dram_read_derived", "dram", "read",
            native["dram_read_bytes"],
            "read_main_memory_datas * (1 - directional L2 read hit rate)",
        ),
    ]
    if native["ub_to_gm_bytes"] is not None:
        specifications.append((
            "traffic:aiv_mte3_ub_to_gm", "mte3", "write",
            native["ub_to_gm_bytes"], "UB_to_GM_datas",
        ))
    rows = []
    for component, role, access_kind, observed, metric in specifications:
        if observed is None:
            continue
        row = _directional_traffic_row(
            task=task,
            mechanism=mechanism,
            component=component,
            resource_role=role,
            access_kind=access_kind,
            observed=float(observed),
            provider="ascend_msprof",
            metric=metric,
        )
        if row is not None:
            rows.append(row)
    return rows


def _metax_directional_traffic_rows(
        *, task: Mapping[str, Any], mechanism: Mapping[str, Any],
        provenance: Mapping[str, Any],
        unavailable: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Compare C500 traffic only at matching directional interfaces."""

    native = mechanism["observation"]["native"]
    if native["backend"] != "metax_mcprofiler":
        raise ValueError(
            "MetaX directional traffic requires mcProfiler observation")
    analytical = mechanism["analytical_input"]
    specifications = (
        (
            "traffic:l2c_source_read",
            "l2_cache",
            "read",
            "l2c_memory_access_bytes",
            "demand_per_resource.xbar",
            "mcProfiler l2c_memacs read-side request bytes",
        ),
        (
            "traffic:downstream_global_read",
            "dram",
            "read",
            "global_read_bytes",
            "demand_per_resource.dram",
            "mcProfiler Global Memory Read bytes",
        ),
        (
            "traffic:downstream_global_write",
            "dram",
            "write",
            "global_write_bytes",
            "demand_per_resource.dram",
            "mcProfiler Global Memory Write bytes",
        ),
    )
    comparison_evidence = _canonical_capture_comparison_evidence(
        mechanism["hardware_id"])
    rows = []
    for (component, resource_role, access_kind, observed_field,
         provenance_path, metric) in specifications:
        if provenance_path not in provenance or provenance_path in unavailable:
            continue
        source = provenance[provenance_path]
        if source["kind"] not in PROFILER_EVIDENCE_KINDS:
            continue
        if source["uses_target_duration"]:
            raise ValueError(
                f"component evidence {provenance_path!r} for "
                f"{task['case_id']!r} consumes target duration")
        observed_value = native[observed_field]
        if observed_value is None:
            raise ValueError(
                f"MetaX directional observation {observed_field!r} is absent "
                "despite profiler provenance")
        observed = float(observed_value)
        if not math.isfinite(observed) or observed < 0:
            raise ValueError(
                f"MetaX directional observation must be finite and "
                f"non-negative: {observed!r}")
        predicted = _memory_path_bytes(
            analytical,
            access_kind=access_kind,
            resource_role=resource_role,
        )
        if predicted == 0 and observed == 0:
            continue
        rows.append({
            "hardware_id": mechanism["hardware_id"],
            "family": task["family"],
            "case_id": task["case_id"],
            "kernel_ordinal": mechanism["kernel_ordinal"],
            "variant": mechanism["variant"],
            "kernel_name": mechanism["kernel_name"],
            "actual_ns": mechanism["timing_target"]["duration_ns"],
            "timing_source": mechanism["timing_target"]["source"],
            "component": component,
            "unit": "bytes",
            "predicted": predicted,
            "observed": observed,
            **comparison_evidence,
            "provenance_kind": source["kind"],
            "provider": source["provider"],
            "metric": metric,
        })
    return rows


class _MechanismHardwareContract(NamedTuple):
    """What one hardware contributes to the mechanism-validation comparison.

    ``directional_traffic_rows`` produces that hardware's directional traffic
    rows, or None when that hardware's rows depend on the oracle fields and
    are appended separately where the comparison is built (the MetaX case).
    ``native_profiler_op_type`` is True when the retained
    observation carries the device profiler's native op type, which the
    report displays; the other backends leave it unset.
    """

    directional_traffic_rows: object = None
    native_profiler_op_type: bool = False


_MECHANISM_HARDWARE = {
    "nvidia_a800": _MechanismHardwareContract(
        directional_traffic_rows=_nvidia_directional_traffic_rows),
    "metax_c500": _MechanismHardwareContract(),
    "ascend_910b": _MechanismHardwareContract(
        directional_traffic_rows=_ascend_directional_traffic_rows,
        native_profiler_op_type=True),
}


def _comparison_rows(documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        comparison_evidence = _canonical_capture_comparison_evidence(
            hardware_id)
        for task in document["tasks"]:
            for kernel in task["kernels"]:
                mechanism = kernel["mechanisms"]
                analytical = mechanism["analytical_input"]
                if hardware_id not in _MECHANISM_HARDWARE:
                    raise ValueError(
                        "unknown mechanism-validation hardware: "
                        f"{hardware_id!r}")
                directional = _MECHANISM_HARDWARE[
                    hardware_id].directional_traffic_rows
                if directional is not None:
                    rows.extend(directional(task=task, mechanism=mechanism))
                oracle = mechanism["oracle"]
                if not oracle["available"]:
                    continue
                provenance = oracle["field_provenance"]
                unavailable = oracle["unavailable_fields"]
                observed_input = oracle["observed_input"]
                if _MECHANISM_HARDWARE[hardware_id].native_profiler_op_type:
                    profiler_op_type = mechanism[
                        "observation"]["native"]["op_type"]
                else:
                    profiler_op_type = None
                if hardware_id == "metax_c500":
                    rows.extend(_metax_directional_traffic_rows(
                        task=task,
                        mechanism=mechanism,
                        provenance=provenance,
                        unavailable=unavailable,
                    ))
                for role in DEMAND_ROLES:
                    path = f"demand_per_resource.{role}"
                    component = f"demand:{role}"
                    if not _is_semantically_comparable_component(
                            hardware_id, task["family"],
                            mechanism["variant"], component,
                            profiler_op_type):
                        continue
                    if path not in provenance or path in unavailable:
                        continue
                    source = provenance[path]
                    if source["kind"] not in PROFILER_EVIDENCE_KINDS:
                        continue
                    if source["uses_target_duration"]:
                        raise ValueError(
                            f"component evidence {path!r} for "
                            f"{task['case_id']!r} consumes target duration")
                    predicted = analytical["demand_per_resource"][role]
                    observed = observed_input["demand_per_resource"][role]
                    if predicted == 0 and observed == 0:
                        continue
                    if role in {
                            "l1_cache", "smem", "xbar", "l2_cache",
                            "dram", "mte3"}:
                        unit = "bytes"
                    elif role in {
                            "compute_tensor", "compute_vector",
                            "compute_reduction", "compute_fma", "compute_sfu",
                            "compute_alu", "warp_issue"}:
                        unit = "operations"
                    else:
                        raise ValueError(
                            f"unknown demand role unit: {role!r}")
                    rows.append({
                        "hardware_id": hardware_id,
                        "family": task["family"],
                        "case_id": task["case_id"],
                        "kernel_ordinal": mechanism["kernel_ordinal"],
                        "variant": mechanism["variant"],
                        "kernel_name": mechanism["kernel_name"],
                        "actual_ns": mechanism["timing_target"]["duration_ns"],
                        "timing_source": mechanism["timing_target"]["source"],
                        "component": component,
                        "unit": unit,
                        "predicted": predicted,
                        "observed": observed,
                        **comparison_evidence,
                        "provenance_kind": source["kind"],
                        "provider": source["provider"],
                        "metric": source["metric"],
                    })
                for role in TIME_ROLES:
                    path = f"resource_time_overrides_ns.{role}"
                    if path not in provenance or path in unavailable:
                        continue
                    source = provenance[path]
                    if source["kind"] not in PROFILER_EVIDENCE_KINDS:
                        continue
                    if source["uses_target_duration"]:
                        raise ValueError(
                            f"component evidence {path!r} for "
                            f"{task['case_id']!r} consumes target duration")
                    overrides = observed_input["resource_time_overrides_ns"]
                    if role not in overrides:
                        raise RuntimeError(
                            f"provenance declares absent time override {path}")
                    predicted_times = mechanism["forward"][
                        "resource_time_by_role_ns"]
                    if role not in predicted_times:
                        continue
                    if predicted_times[role] == 0 and overrides[role] == 0:
                        continue
                    rows.append({
                        "hardware_id": hardware_id,
                        "family": task["family"],
                        "case_id": task["case_id"],
                        "kernel_ordinal": mechanism["kernel_ordinal"],
                        "variant": mechanism["variant"],
                        "kernel_name": mechanism["kernel_name"],
                        "actual_ns": mechanism["timing_target"]["duration_ns"],
                        "timing_source": mechanism["timing_target"]["source"],
                        "component": f"service_time:{role}",
                        "unit": "ns",
                        "predicted": predicted_times[role],
                        "observed": overrides[role],
                        **comparison_evidence,
                        "provenance_kind": source["kind"],
                        "provider": source["provider"],
                        "metric": source["metric"],
                    })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields_: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields_)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields_})


def _baseline_rows(
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for result_scope, source_rows in (
            ("real_kernel", kernel_rows),
            ("aggregate_task", task_rows)):
        for hardware_id, _ in HARDWARE:
            selected_hardware = [
                row for row in source_rows
                if row["hardware_id"] == hardware_id]
            for family in ("ALL", *FAMILY_LABELS):
                if family == "ALL":
                    selected = selected_hardware
                elif family in FAMILY_LABELS:
                    selected = [
                        row for row in selected_hardware
                        if row["family"] == family]
                else:
                    raise ValueError(
                        f"unknown baseline family: {family!r}")
                statuses = [_comparison_status(row) for row in selected]
                comparable = [
                    row for row in selected
                    if _comparison_status(row) == "comparable"
                ]
                diagnostic_metrics = (
                    _metric_summary([
                        (
                            row["predicted_ns"],
                            _observed_duration_ns(row, result_scope),
                        )
                        for row in selected
                    ])
                    if selected else None
                )
                metrics = (
                    _metric_summary([
                        (
                            row["predicted_ns"],
                            _observed_duration_ns(row, result_scope),
                        )
                        for row in comparable
                    ])
                    if comparable else None
                )
                if result_scope == "real_kernel":
                    task_count = len({row["case_id"] for row in selected})
                    kernel_invocation_count = len(selected)
                elif result_scope == "aggregate_task":
                    task_count = len(selected)
                    kernel_invocation_count = sum(
                        int(row["kernel_count"]) for row in selected)
                else:
                    raise AssertionError(
                        f"unhandled baseline scope: {result_scope!r}")
                support_counts = {
                    level: sum(
                        row["support_level"] == level for row in selected)
                    for level in SUPPORT_LEVELS
                }
                rows.append({
                    "result_scope": result_scope,
                    "hardware_id": hardware_id,
                    "family": family,
                    "row_count": len(selected),
                    "comparable_count": sum(
                        status == "comparable" for status in statuses),
                    "not_comparable_count": sum(
                        status == "not_comparable" for status in statuses),
                    "unobserved_count": sum(
                        status == "unobserved" for status in statuses),
                    "task_count": task_count,
                    "kernel_invocation_count": kernel_invocation_count,
                    "mape_pct": (
                        None if metrics is None else metrics["mape_pct"]),
                    "mdape_pct": (
                        None if metrics is None else metrics["mdape_pct"]),
                    "p50_ape_pct": (
                        None if metrics is None else metrics["p50_ape_pct"]),
                    "p90_ape_pct": (
                        None if metrics is None else metrics["p90_ape_pct"]),
                    "p99_ape_pct": (
                        None if metrics is None else metrics["p99_ape_pct"]),
                    "max_ape_pct": (
                        None if metrics is None else metrics["max_ape_pct"]),
                    "weighted_ape_pct": (
                        None if metrics is None
                        else metrics["weighted_ape_pct"]),
                    "mean_signed_error_pct": (
                        None if metrics is None
                        else metrics["mean_signed_error_pct"]),
                    "median_signed_error_pct": (
                        None if metrics is None
                        else metrics["median_signed_error_pct"]),
                    "diagnostic_mape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["mape_pct"]),
                    "diagnostic_mdape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["mdape_pct"]),
                    "diagnostic_p50_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["p50_ape_pct"]),
                    "diagnostic_p90_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["p90_ape_pct"]),
                    "diagnostic_p99_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["p99_ape_pct"]),
                    "diagnostic_max_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["max_ape_pct"]),
                    "diagnostic_weighted_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["weighted_ape_pct"]),
                    "diagnostic_mean_signed_error_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["mean_signed_error_pct"]),
                    "diagnostic_median_signed_error_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["median_signed_error_pct"]),
                    **{
                        f"{level}_row_count": count
                        for level, count in support_counts.items()
                    },
                })
    return rows


def _read_result_csv(path: Path, required_fields: Sequence[str]) -> list[Mapping[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"comparison result is missing: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"comparison result has no header: {path}")
        missing = sorted(set(required_fields) - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"comparison result {path} is missing fields: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"comparison result has no rows: {path}")
    return rows


def _verify_manifest_artifact(
        results_dir: Path,
        manifest: Mapping[str, Any],
        relative_path: str) -> None:
    """Fail closed when a comparison input is not bound by its manifest."""

    artifact_hashes = manifest["artifact_sha256"]
    if not isinstance(artifact_hashes, Mapping):
        raise TypeError("before-results artifact_sha256 must be a mapping")
    if relative_path not in artifact_hashes:
        raise ValueError(
            "before-results manifest does not bind comparison artifact: "
            f"{relative_path}")
    expected = artifact_hashes[relative_path]
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(
            "before-results manifest has an invalid SHA256 for "
            f"{relative_path}: {expected!r}")
    path = results_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"comparison result is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            "before-results comparison artifact SHA256 mismatch for "
            f"{relative_path}: expected {expected}, got {actual}")


def _result_join_key(
        row: Mapping[str, Any], result_scope: str) -> tuple[Any, ...]:
    if result_scope == "real_kernel":
        ordinal = int(row["kernel_ordinal"])
        if ordinal < 0:
            raise ValueError("comparison kernel ordinal must be non-negative")
        return (
            result_scope,
            row["hardware_id"],
            row["family"],
            row["op_type"],
            row["case_id"],
            ordinal,
        )
    elif result_scope == "aggregate_task":
        return (
            result_scope,
            row["hardware_id"],
            row["family"],
            row["case_id"],
        )
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")


def _index_result_rows(
        rows: Sequence[Mapping[str, Any]],
        result_scope: str) -> Mapping[tuple[Any, ...], Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        if "result_scope" in row and row["result_scope"] != result_scope:
            raise ValueError(
                f"comparison row scope {row['result_scope']!r} does not "
                f"match {result_scope!r}")
        key = _result_join_key(row, result_scope)
        if key in indexed:
            raise ValueError(f"duplicate comparison row identity: {key!r}")
        indexed[key] = row
    return indexed


def _require_same_result_identity(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        fields_: Sequence[str],
        key: tuple[Any, ...]) -> None:
    for field in fields_:
        before_value = before[field]
        after_value = after[field]
        if str(before_value) != str(after_value):
            raise ValueError(
                f"comparison identity {key!r} disagrees on {field}: "
                f"{before_value!r} != {after_value!r}")


def _comparison_model_identity(
        row: Mapping[str, Any], result_scope: str) -> str:
    if result_scope == "real_kernel":
        fields_ = (
            "kernel_kind", "variant", "launch_algorithm",
            "launch_grid_size", "launch_block_size",
            "launch_execution_variant", "launch_tile_shape",
            "launch_num_kv_splits",
        )
    elif result_scope == "aggregate_task":
        fields_ = ("kernel_count", "task_execution_signature")
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")
    return json.dumps(
        {field: str(row[field]) for field in fields_},
        separators=(",", ":"),
        sort_keys=True,
    )


def _positive_csv_float(
        row: Mapping[str, Any], field: str, key: tuple[Any, ...]) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"comparison identity {key!r} has invalid {field}: {value!r}")
    return value


def _before_after_case_rows(
        before_kernel_rows: Sequence[Mapping[str, Any]],
        before_task_rows: Sequence[Mapping[str, Any]],
        after_kernel_rows: Sequence[Mapping[str, Any]],
        after_task_rows: Sequence[Mapping[str, Any]],
        ) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Join historical predictions onto current frozen physical buckets."""

    result = []
    scope_audits = []
    for result_scope, before_rows, after_rows in (
            ("real_kernel", before_kernel_rows, after_kernel_rows),
            ("aggregate_task", before_task_rows, after_task_rows)):
        before_index = _index_result_rows(before_rows, result_scope)
        after_index = _index_result_rows(after_rows, result_scope)
        before_keys = set(before_index)
        after_keys = set(after_index)
        missing_before = sorted(after_keys - before_keys)
        missing_after = sorted(before_keys - after_keys)
        if missing_before or missing_after:
            raise ValueError(
                f"{result_scope} comparison row sets differ: "
                f"missing_before={missing_before[:5]!r}, "
                f"missing_after={missing_after[:5]!r}")
        if result_scope == "real_kernel":
            identity_fields = (
                "hardware_id", "family", "op_type", "case_id",
                "kernel_ordinal", "kernel_name",
            )
            measured_field = "actual_ns"
        elif result_scope == "aggregate_task":
            identity_fields = (
                "hardware_id", "family", "case_id",
                "kernel_count", "gemm_m", "gemm_n", "gemm_k",
                "query_length", "kv_length", "executed_attention_splits",
                "attention_layout", "moe_experts", "moe_topk",
                "moe_max_tokens_per_expert", "vector_tokens",
            )
            measured_field = "observed_ns"
        else:
            raise AssertionError(
                f"unhandled before/after scope: {result_scope!r}")
        for key in sorted(after_index):
            before = before_index[key]
            after = after_index[key]
            _require_same_result_identity(
                before, after, identity_fields, key)
            before_measured = _positive_csv_float(
                before, measured_field, key)
            after_measured = _positive_csv_float(after, measured_field, key)
            if before_measured != after_measured:
                raise ValueError(
                    f"comparison identity {key!r} changed {measured_field}: "
                    f"{before_measured!r} != {after_measured!r}")
            before_predicted = _positive_csv_float(
                before, "predicted_ns", key)
            after_predicted = _positive_csv_float(
                after, "predicted_ns", key)
            before_signed = (
                (before_predicted - after_measured) / after_measured * 100.0)
            after_signed = (
                (after_predicted - after_measured) / after_measured * 100.0)
            before_absolute = abs(before_signed)
            after_absolute = abs(after_signed)
            comparison_status = _comparison_status(after)
            comparison_gate_passed = comparison_status == "comparable"
            if result_scope == "real_kernel":
                kernel_ordinal = int(after["kernel_ordinal"])
                kernel_kind = after["kernel_kind"]
                before_variant_or_execution_graph = before["variant"]
                variant_or_execution_graph = after["variant"]
            elif result_scope == "aggregate_task":
                kernel_ordinal = None
                kernel_kind = None
                before_variant_or_execution_graph = before[
                    "task_execution_signature"]
                variant_or_execution_graph = after[
                    "task_execution_signature"]
            else:
                raise AssertionError(
                    f"unhandled before/after scope: {result_scope!r}")
            before_model_identity = _comparison_model_identity(
                before, result_scope)
            after_model_identity = _comparison_model_identity(
                after, result_scope)
            result.append({
                "result_scope": result_scope,
                "hardware_id": after["hardware_id"],
                "family": after["family"],
                "case_id": after["case_id"],
                "kernel_ordinal": kernel_ordinal,
                "kernel_kind": kernel_kind,
                "before_variant_or_execution_graph": (
                    before_variant_or_execution_graph),
                "variant_or_execution_graph": variant_or_execution_graph,
                "before_model_identity": before_model_identity,
                "after_model_identity": after_model_identity,
                "model_identity_changed": (
                    before_model_identity != after_model_identity),
                "timing_source": after["timing_source"],
                "measured_ns": after_measured,
                "before_predicted_ns": before_predicted,
                "after_predicted_ns": after_predicted,
                "prediction_change_ns": after_predicted - before_predicted,
                "before_signed_error_pct": before_signed,
                "after_signed_error_pct": after_signed,
                "before_absolute_error_pct": before_absolute,
                "after_absolute_error_pct": after_absolute,
                "ape_change_pct_points": after_absolute - before_absolute,
                "before_absolute_duration_error_ns": abs(
                    before_predicted - after_measured),
                "after_absolute_duration_error_ns": abs(
                    after_predicted - after_measured),
                "absolute_duration_error_change_ns": (
                    abs(after_predicted - after_measured)
                    - abs(before_predicted - after_measured)),
                "before_case_gate_passed": (
                    comparison_gate_passed
                    and before_absolute <= CASE_APE_LIMIT_PCT),
                "after_case_gate_passed": (
                    comparison_gate_passed
                    and after_absolute <= CASE_APE_LIMIT_PCT),
                "before_support_level": before["support_level"],
                "after_support_level": after["support_level"],
                "support_level_changed": (
                    before["support_level"] != after["support_level"]),
                "bucket_schema": after["bucket_schema"],
                "physical_bucket_id": after["physical_bucket_id"],
                "physical_bucket_short_id": after[
                    "physical_bucket_short_id"],
                "bucket_dimensions": after["bucket_dimensions"],
                "bucket_basis": after["bucket_basis"],
                "comparison_status": comparison_status,
                "comparison_reason": after["comparison_reason"],
            })
        scope_audits.append({
            "result_scope": result_scope,
            "before_row_count": len(before_rows),
            "after_row_count": len(after_rows),
            "joined_row_count": len(after_keys),
            "before_duplicate_key_count": 0,
            "after_duplicate_key_count": 0,
            "missing_before_count": 0,
            "missing_after_count": 0,
            "identity_mismatch_count": 0,
            "measurement_mismatch_count": 0,
            "model_identity_change_count": sum(
                row["model_identity_changed"]
                for row in result
                if row["result_scope"] == result_scope),
        })
    return result, {"scopes": scope_audits}


def _before_after_bucket_rows(
        case_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in case_rows:
        key = (row["result_scope"], row["physical_bucket_id"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    result = []
    for key in sorted(groups):
        result_scope, bucket_id = key
        values = groups[key]
        first = values[0]
        for row in values[1:]:
            for field in (
                    "hardware_id", "bucket_schema", "bucket_dimensions",
                    "bucket_basis", "physical_bucket_short_id"):
                if row[field] != first[field]:
                    raise ValueError(
                        f"before/after bucket {key!r} disagrees on {field}")
        families = sorted({row["family"] for row in values})
        statuses = [_comparison_status(row) for row in values]
        status_counts = {
            status: sum(value == status for value in statuses)
            for status in COMPARISON_STATUSES
        }
        comparison_coverage_passed = (
            status_counts["comparable"] == len(values))
        before_metrics = _metric_summary([
            (float(row["before_predicted_ns"]), float(row["measured_ns"]))
            for row in values
        ])
        after_metrics = _metric_summary([
            (float(row["after_predicted_ns"]), float(row["measured_ns"]))
            for row in values
        ])
        before_mape_passed = (
            comparison_coverage_passed
            and before_metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        after_mape_passed = (
            comparison_coverage_passed
            and after_metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        before_max_passed = (
            comparison_coverage_passed
            and before_metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        after_max_passed = (
            comparison_coverage_passed
            and after_metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        before_diagnostic_mape_passed = (
            before_metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        after_diagnostic_mape_passed = (
            after_metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        before_diagnostic_max_passed = (
            before_metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        after_diagnostic_max_passed = (
            after_metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        result.append({
            "result_scope": result_scope,
            "hardware_id": first["hardware_id"],
            "family": first["family"] if len(families) == 1 else "mixed",
            "physical_bucket_id": bucket_id,
            "physical_bucket_short_id": first["physical_bucket_short_id"],
            "bucket_schema": first["bucket_schema"],
            "bucket_dimensions": first["bucket_dimensions"],
            "bucket_basis": first["bucket_basis"],
            "row_count": len(values),
            "comparable_count": status_counts["comparable"],
            "not_comparable_count": status_counts["not_comparable"],
            "unobserved_count": status_counts["unobserved"],
            "comparison_coverage_pct": (
                status_counts["comparable"] / len(values) * 100.0),
            "comparison_coverage_gate_passed": (
                comparison_coverage_passed),
            "before_mape_pct": before_metrics["mape_pct"],
            "after_mape_pct": after_metrics["mape_pct"],
            "mape_change_pct_points": (
                after_metrics["mape_pct"] - before_metrics["mape_pct"]),
            "before_mdape_pct": before_metrics["mdape_pct"],
            "after_mdape_pct": after_metrics["mdape_pct"],
            "before_p90_ape_pct": before_metrics["p90_ape_pct"],
            "after_p90_ape_pct": after_metrics["p90_ape_pct"],
            "before_p99_ape_pct": before_metrics["p99_ape_pct"],
            "after_p99_ape_pct": after_metrics["p99_ape_pct"],
            "before_max_ape_pct": before_metrics["max_ape_pct"],
            "after_max_ape_pct": after_metrics["max_ape_pct"],
            "max_ape_change_pct_points": (
                after_metrics["max_ape_pct"]
                - before_metrics["max_ape_pct"]),
            "before_mean_signed_error_pct": before_metrics[
                "mean_signed_error_pct"],
            "after_mean_signed_error_pct": after_metrics[
                "mean_signed_error_pct"],
            "before_median_signed_error_pct": before_metrics[
                "median_signed_error_pct"],
            "after_median_signed_error_pct": after_metrics[
                "median_signed_error_pct"],
            "before_case_violation_count": sum(
                float(row["before_absolute_error_pct"])
                > CASE_APE_LIMIT_PCT for row in values),
            "after_case_violation_count": sum(
                float(row["after_absolute_error_pct"])
                > CASE_APE_LIMIT_PCT for row in values),
            "before_diagnostic_mape_gate_passed": (
                before_diagnostic_mape_passed),
            "after_diagnostic_mape_gate_passed": (
                after_diagnostic_mape_passed),
            "before_diagnostic_max_ape_gate_passed": (
                before_diagnostic_max_passed),
            "after_diagnostic_max_ape_gate_passed": (
                after_diagnostic_max_passed),
            "before_diagnostic_accuracy_gate_passed": (
                before_diagnostic_mape_passed
                and before_diagnostic_max_passed),
            "after_diagnostic_accuracy_gate_passed": (
                after_diagnostic_mape_passed
                and after_diagnostic_max_passed),
            "before_mape_gate_passed": before_mape_passed,
            "after_mape_gate_passed": after_mape_passed,
            "before_max_ape_gate_passed": before_max_passed,
            "after_max_ape_gate_passed": after_max_passed,
            "before_accuracy_gate_passed": (
                before_mape_passed and before_max_passed),
            "after_accuracy_gate_passed": (
                after_mape_passed and after_max_passed),
        })
    if not result:
        raise ValueError("before/after bucket report has no rows")
    return result


def _before_after_summary_rows(
        case_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = []
    for result_scope in ("real_kernel", "aggregate_task"):
        scope_rows = [
            row for row in case_rows
            if row["result_scope"] == result_scope]
        for hardware_id, _ in HARDWARE:
            hardware_rows = [
                row for row in scope_rows
                if row["hardware_id"] == hardware_id]
            if not hardware_rows:
                if result_scope == "aggregate_task":
                    raise ValueError(
                        "before/after summary has no rows for "
                        f"{result_scope}/{hardware_id}/ALL")
                # Canonical per-kernel timings are intentionally sparse: at
                # present only A800 non-Vector datasets expose real-kernel
                # rows.  Report only measured scope/family combinations rather
                # than fabricating empty summaries for OP-timed datasets.
                continue
            for family in ("ALL", *FAMILY_LABELS):
                if family == "ALL":
                    values = hardware_rows
                elif family in FAMILY_LABELS:
                    values = [
                        row for row in hardware_rows
                        if row["family"] == family
                    ]
                else:
                    raise ValueError(
                        f"unknown before/after summary family: {family!r}")
                if not values:
                    if result_scope == "real_kernel":
                        continue
                    raise ValueError(
                        "before/after summary has no rows for "
                        f"{result_scope}/{hardware_id}/{family}")
                before_metrics = _metric_summary([
                    (
                        float(row["before_predicted_ns"]),
                        float(row["measured_ns"]),
                    )
                    for row in values
                ])
                after_metrics = _metric_summary([
                    (
                        float(row["after_predicted_ns"]),
                        float(row["measured_ns"]),
                    )
                    for row in values
                ])
                # This table is explicitly a provenance-unverified numerical
                # diagnostic.  Count the configured APE condition itself here;
                # comparison eligibility is reported independently and must
                # not be mislabeled as a numerical error violation.
                before_pass_count = sum(
                    float(row["before_absolute_error_pct"])
                    <= CASE_APE_LIMIT_PCT for row in values)
                after_pass_count = sum(
                    float(row["after_absolute_error_pct"])
                    <= CASE_APE_LIMIT_PCT for row in values)
                result.append({
                    "result_scope": result_scope,
                    "hardware_id": hardware_id,
                    "family": family,
                    "row_count": len(values),
                    "before_mape_pct": before_metrics["mape_pct"],
                    "after_mape_pct": after_metrics["mape_pct"],
                    "mape_change_pct_points": (
                        after_metrics["mape_pct"]
                        - before_metrics["mape_pct"]),
                    "before_mdape_pct": before_metrics["mdape_pct"],
                    "after_mdape_pct": after_metrics["mdape_pct"],
                    "before_p90_ape_pct": before_metrics["p90_ape_pct"],
                    "after_p90_ape_pct": after_metrics["p90_ape_pct"],
                    "before_p99_ape_pct": before_metrics["p99_ape_pct"],
                    "after_p99_ape_pct": after_metrics["p99_ape_pct"],
                    "before_max_ape_pct": before_metrics["max_ape_pct"],
                    "after_max_ape_pct": after_metrics["max_ape_pct"],
                    "before_mean_signed_error_pct": before_metrics[
                        "mean_signed_error_pct"],
                    "after_mean_signed_error_pct": after_metrics[
                        "mean_signed_error_pct"],
                    "before_median_signed_error_pct": before_metrics[
                        "median_signed_error_pct"],
                    "after_median_signed_error_pct": after_metrics[
                        "median_signed_error_pct"],
                    "before_case_pass_count": before_pass_count,
                    "after_case_pass_count": after_pass_count,
                    "before_case_violation_count": (
                        len(values) - before_pass_count),
                    "after_case_violation_count": (
                        len(values) - after_pass_count),
                })
    return result


def _bucket_rows(
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in (*kernel_rows, *task_rows):
        key = (row["result_scope"], row["physical_bucket_id"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    result = []
    for key in sorted(groups):
        result_scope, bucket_id = key
        values = groups[key]
        first = values[0]
        for row in values[1:]:
            for field in (
                    "bucket_schema", "hardware_id",
                    "bucket_basis", "bucket_dimensions"):
                if row[field] != first[field]:
                    raise RuntimeError(
                        f"physical bucket {bucket_id!r} disagrees on {field}")
        task_family_contexts = sorted({row["family"] for row in values})
        statuses = [_comparison_status(row) for row in values]
        status_counts = {
            status: sum(value == status for value in statuses)
            for status in COMPARISON_STATUSES
        }
        comparable_values = [
            row for row in values
            if _comparison_status(row) == "comparable"
        ]
        diagnostic_pairs = [
            (float(row["predicted_ns"]),
             _observed_duration_ns(row, result_scope))
            for row in values
        ]
        diagnostic_metrics = _metric_summary(diagnostic_pairs)
        metrics = (
            _metric_summary([
                (float(row["predicted_ns"]),
                 _observed_duration_ns(row, result_scope))
                for row in comparable_values
            ])
            if comparable_values else None
        )
        support_counts = {
            level: sum(row["support_level"] == level for row in values)
            for level in SUPPORT_LEVELS
        }
        unknown = {
            row["support_level"] for row in values
            if row["support_level"] not in SUPPORT_LEVELS
        }
        if unknown:
            raise ValueError(
                f"unknown Forward support levels: {sorted(unknown)}")
        complete_support_pct = (
            support_counts["complete"] / len(values) * 100.0)
        comparison_coverage_pct = (
            len(comparable_values) / len(values) * 100.0)
        comparison_coverage_passed = len(comparable_values) == len(values)
        mape_passed = (
            metrics is not None
            and metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        max_ape_passed = (
            metrics is not None
            and metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        support_passed = complete_support_pct == 100.0
        diagnostic_worst = max(
            values,
            key=lambda row: (
                float(row["absolute_error_pct"]),
                row["case_id"],
                _kernel_ordinal_sort_value(row, result_scope),
            ),
        )
        worst = (
            max(
                comparable_values,
                key=lambda row: (
                    float(row["absolute_error_pct"]),
                    row["case_id"],
                    _kernel_ordinal_sort_value(row, result_scope),
                ),
            )
            if comparable_values else None
        )
        worst_kernel_ordinal = (
            None if worst is None
            else _kernel_ordinal_for_scope(worst, result_scope))
        result.append({
            "bucket_schema": first["bucket_schema"],
            "result_scope": result_scope,
            "hardware_id": first["hardware_id"],
            "family": (
                first["family"] if len(task_family_contexts) == 1
                else "mixed"),
            "task_family_contexts": json.dumps(
                task_family_contexts, separators=(",", ":")),
            "physical_bucket_id": bucket_id,
            "physical_bucket_short_id": first["physical_bucket_short_id"],
            "bucket_dimensions": first["bucket_dimensions"],
            "bucket_basis": first["bucket_basis"],
            "row_count": len(values),
            "comparable_count": status_counts["comparable"],
            "not_comparable_count": status_counts["not_comparable"],
            "unobserved_count": status_counts["unobserved"],
            "comparison_coverage_pct": comparison_coverage_pct,
            "comparison_coverage_gate_passed": comparison_coverage_passed,
            "sample_evidence": (
                "evidence_low_n" if len(values) < 3 else "multi_case"),
            "captured_prediction_count": len(values),
            "complete_support_count": support_counts["complete"],
            "complete_support_pct": complete_support_pct,
            **{
                f"{level}_count": support_counts[level]
                for level in SUPPORT_LEVELS
            },
            "mape_pct": None if metrics is None else metrics["mape_pct"],
            "mdape_pct": None if metrics is None else metrics["mdape_pct"],
            "p90_ape_pct": (
                None if metrics is None else metrics["p90_ape_pct"]),
            "p99_ape_pct": (
                None if metrics is None else metrics["p99_ape_pct"]),
            "max_ape_pct": (
                None if metrics is None else metrics["max_ape_pct"]),
            "mean_signed_error_pct": (
                None if metrics is None
                else metrics["mean_signed_error_pct"]),
            "median_signed_error_pct": (
                None if metrics is None
                else metrics["median_signed_error_pct"]),
            "diagnostic_mape_pct": diagnostic_metrics["mape_pct"],
            "diagnostic_mdape_pct": diagnostic_metrics["mdape_pct"],
            "diagnostic_p90_ape_pct": diagnostic_metrics["p90_ape_pct"],
            "diagnostic_p99_ape_pct": diagnostic_metrics["p99_ape_pct"],
            "diagnostic_max_ape_pct": diagnostic_metrics["max_ape_pct"],
            "diagnostic_mean_signed_error_pct": diagnostic_metrics[
                "mean_signed_error_pct"],
            "diagnostic_median_signed_error_pct": diagnostic_metrics[
                "median_signed_error_pct"],
            "worst_case_id": None if worst is None else worst["case_id"],
            "worst_kernel_ordinal": worst_kernel_ordinal,
            "diagnostic_worst_case_id": diagnostic_worst["case_id"],
            "diagnostic_worst_kernel_ordinal": _kernel_ordinal_for_scope(
                diagnostic_worst, result_scope),
            "mape_gate_passed": mape_passed,
            "max_ape_gate_passed": max_ape_passed,
            "accuracy_bucket_gate_passed": (
                comparison_coverage_passed
                and mape_passed and max_ape_passed),
            "support_gate_passed": support_passed,
            "bucket_gate_passed": (
                comparison_coverage_passed
                and mape_passed and max_ape_passed and support_passed),
        })
    if not result:
        raise RuntimeError("physical bucket report produced no rows")
    return result


def _severe_duration_rows(
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]],
        threshold_pct: float = 50.0) -> list[Mapping[str, Any]]:
    if not math.isfinite(threshold_pct) or threshold_pct <= 0:
        raise ValueError("severe duration threshold must be positive")
    result = []
    for row in kernel_rows:
        if row["absolute_error_pct"] < threshold_pct:
            continue
        result.append({
            "result_scope": row["result_scope"],
            "hardware_id": row["hardware_id"],
            "family": row["family"],
            "case_id": row["case_id"],
            "kernel_ordinal": row["kernel_ordinal"],
            "kernel_kind": row["kernel_kind"],
            "variant": row["variant"],
            "kernel_count": 1,
            "timing_source": row["timing_source"],
            "predicted_ns": row["predicted_ns"],
            "observed_ns": row["actual_ns"],
            "absolute_duration_error_ns": abs(
                row["predicted_ns"] - row["actual_ns"]),
            "signed_error_pct": row["signed_error_pct"],
            "absolute_error_pct": row["absolute_error_pct"],
            "physical_bucket_id": row["physical_bucket_id"],
            "comparison_status": _comparison_status(row),
            "comparison_reason": row["comparison_reason"],
        })
    for row in task_rows:
        if row["absolute_error_pct"] < threshold_pct:
            continue
        result.append({
            "result_scope": row["result_scope"],
            "hardware_id": row["hardware_id"],
            "family": row["family"],
            "case_id": row["case_id"],
            "kernel_ordinal": None,
            "kernel_kind": None,
            "variant": row["variant_signature"],
            "kernel_count": row["kernel_count"],
            "timing_source": row["timing_source"],
            "predicted_ns": row["predicted_ns"],
            "observed_ns": row["observed_ns"],
            "absolute_duration_error_ns": abs(
                row["predicted_ns"] - row["observed_ns"]),
            "signed_error_pct": row["signed_error_pct"],
            "absolute_error_pct": row["absolute_error_pct"],
            "physical_bucket_id": row["physical_bucket_id"],
            "comparison_status": _comparison_status(row),
            "comparison_reason": row["comparison_reason"],
        })
    result.sort(key=lambda row: (
        -float(row["observed_ns"]),
        row["result_scope"],
        row["hardware_id"],
        row["case_id"],
        -1 if row["kernel_ordinal"] is None else int(row["kernel_ordinal"]),
    ))
    return result


def _case_impact_rows(
        bucket_rows: Sequence[Mapping[str, Any]],
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]],
        threshold_pct: float = CASE_APE_LIMIT_PCT,
        ) -> list[Mapping[str, Any]]:
    """Rank diagnostic row discrepancies by absolute time impact."""

    if not math.isfinite(threshold_pct) or threshold_pct <= 0:
        raise ValueError("case-impact threshold must be positive")
    bucket_index = {}
    for bucket in bucket_rows:
        key = (bucket["result_scope"], bucket["physical_bucket_id"])
        if key in bucket_index:
            raise RuntimeError(
                f"duplicate physical bucket in case-impact join: {key!r}")
        bucket_index[key] = bucket

    candidates = []
    for result_scope, source_rows in (
            ("real_kernel", kernel_rows),
            ("aggregate_task", task_rows)):
        for row in source_rows:
            absolute_error_pct = float(row["absolute_error_pct"])
            if absolute_error_pct <= threshold_pct:
                continue
            bucket_key = (result_scope, row["physical_bucket_id"])
            if bucket_key not in bucket_index:
                raise RuntimeError(
                    "case-impact row has no matching physical bucket: "
                    f"{bucket_key!r}")
            bucket = bucket_index[bucket_key]
            measured_ns = _observed_duration_ns(row, result_scope)
            predicted_ns = float(row["predicted_ns"])
            if result_scope == "real_kernel":
                measured_duration_field = "actual_ns"
                kernel_kind = row["kernel_kind"]
                variant_or_execution_graph = row["variant"]
                kernel_count = 1
            elif result_scope == "aggregate_task":
                measured_duration_field = "observed_ns"
                kernel_kind = None
                variant_or_execution_graph = row["task_execution_signature"]
                kernel_count = int(row["kernel_count"])
            else:
                raise AssertionError(
                    f"unhandled case-impact scope: {result_scope!r}")
            candidates.append({
                "result_scope": result_scope,
                "hardware_id": row["hardware_id"],
                "family": row["family"],
                "case_id": row["case_id"],
                "kernel_ordinal": _kernel_ordinal_for_scope(
                    row, result_scope),
                "kernel_kind": kernel_kind,
                "variant_or_execution_graph": variant_or_execution_graph,
                "kernel_count": kernel_count,
                "timing_source": row["timing_source"],
                "measured_duration_field": measured_duration_field,
                "measured_ns": measured_ns,
                "predicted_ns": predicted_ns,
                "absolute_duration_error_ns": abs(
                    predicted_ns - measured_ns),
                "signed_error_pct": float(row["signed_error_pct"]),
                "absolute_error_pct": absolute_error_pct,
                "case_ape_gate_passed": False,
                "physical_bucket_id": row["physical_bucket_id"],
                "bucket_mape_pct": float(bucket["diagnostic_mape_pct"]),
                "bucket_mape_gate_passed": bool(
                    bucket["diagnostic_mape_pct"]
                    <= BUCKET_MAPE_LIMIT_PCT),
                "bucket_max_ape_pct": float(
                    bucket["diagnostic_max_ape_pct"]),
                "bucket_max_ape_gate_passed": bool(
                    bucket["diagnostic_max_ape_pct"]
                    <= CASE_APE_LIMIT_PCT),
                "support_level": row["support_level"],
                "comparison_status": _comparison_status(row),
                "comparison_reason": row["comparison_reason"],
            })
    candidates.sort(key=lambda row: (
        -float(row["absolute_duration_error_ns"]),
        -float(row["absolute_error_pct"]),
        -float(row["measured_ns"]),
        row["result_scope"],
        row["hardware_id"],
        row["case_id"],
        -1 if row["kernel_ordinal"] is None else int(row["kernel_ordinal"]),
    ))
    return [
        {"impact_rank": rank, **row}
        for rank, row in enumerate(candidates, start=1)
    ]


def _accuracy_gate(
        bucket_rows: Sequence[Mapping[str, Any]],
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    scopes = []
    all_case_violations = []
    all_accuracy_bucket_violations = []
    all_support_bucket_violations = []
    all_comparison_bucket_violations = []
    all_comparison_row_violations = []
    all_goal_bucket_violations = []
    for result_scope, rows in (
            ("real_kernel", kernel_rows),
            ("aggregate_task", task_rows)):
        selected_buckets = [
            row for row in bucket_rows
            if row["result_scope"] == result_scope
        ]
        if not rows or not selected_buckets:
            raise RuntimeError(
                f"accuracy gate has no rows for {result_scope}")
        comparable_rows = [
            row for row in rows
            if _comparison_status(row) == "comparable"
        ]
        diagnostic_metrics = _metric_summary([
            (
                float(row["predicted_ns"]),
                _observed_duration_ns(row, result_scope),
            )
            for row in rows
        ])
        metrics = (
            _metric_summary([
                (
                    float(row["predicted_ns"]),
                    _observed_duration_ns(row, result_scope),
                )
                for row in comparable_rows
            ])
            if comparable_rows else None
        )
        comparison_row_violations = [
            {
                "result_scope": result_scope,
                "hardware_id": row["hardware_id"],
                "family": row["family"],
                "case_id": row["case_id"],
                "kernel_ordinal": _kernel_ordinal_for_scope(
                    row, result_scope),
                "comparison_status": _comparison_status(row),
                "comparison_reason": row["comparison_reason"],
                "physical_bucket_id": row["physical_bucket_id"],
            }
            for row in rows
            if _comparison_status(row) != "comparable"
        ]
        case_violations = [
            {
                "result_scope": result_scope,
                "hardware_id": row["hardware_id"],
                "family": row["family"],
                "case_id": row["case_id"],
                "kernel_ordinal": _kernel_ordinal_for_scope(
                    row, result_scope),
                "absolute_error_pct": row["absolute_error_pct"],
                "physical_bucket_id": row["physical_bucket_id"],
            }
            for row in comparable_rows
            if row["absolute_error_pct"] > CASE_APE_LIMIT_PCT
        ]
        goal_bucket_violations = [
            {
                "result_scope": result_scope,
                "physical_bucket_id": row["physical_bucket_id"],
                "mape_pct": row["mape_pct"],
                "max_ape_pct": row["max_ape_pct"],
                "diagnostic_mape_pct": row["diagnostic_mape_pct"],
                "diagnostic_max_ape_pct": row[
                    "diagnostic_max_ape_pct"],
                "comparison_coverage_pct": row[
                    "comparison_coverage_pct"],
                "comparison_coverage_gate_passed": row[
                    "comparison_coverage_gate_passed"],
                "complete_support_pct": row["complete_support_pct"],
                "mape_gate_passed": row["mape_gate_passed"],
                "max_ape_gate_passed": row["max_ape_gate_passed"],
                "support_gate_passed": row["support_gate_passed"],
            }
            for row in selected_buckets
            if not row["bucket_gate_passed"]
        ]
        accuracy_bucket_violations = [
            row for row in goal_bucket_violations
            if row["comparison_coverage_gate_passed"]
            and (not row["mape_gate_passed"]
                 or not row["max_ape_gate_passed"])
        ]
        comparison_bucket_violations = [
            row for row in goal_bucket_violations
            if not row["comparison_coverage_gate_passed"]
        ]
        support_bucket_violations = [
            row for row in goal_bucket_violations
            if not row["support_gate_passed"]
        ]
        all_case_violations.extend(case_violations)
        all_accuracy_bucket_violations.extend(accuracy_bucket_violations)
        all_support_bucket_violations.extend(support_bucket_violations)
        all_comparison_bucket_violations.extend(
            comparison_bucket_violations)
        all_comparison_row_violations.extend(comparison_row_violations)
        all_goal_bucket_violations.extend(goal_bucket_violations)
        comparison_coverage_passed = not comparison_row_violations
        diagnostic_case_violation_count = sum(
            float(row["absolute_error_pct"]) > CASE_APE_LIMIT_PCT
            for row in rows
        )
        diagnostic_bucket_mape_violation_count = sum(
            float(row["diagnostic_mape_pct"]) > BUCKET_MAPE_LIMIT_PCT
            for row in selected_buckets
        )
        diagnostic_bucket_max_ape_violation_count = sum(
            float(row["diagnostic_max_ape_pct"]) > CASE_APE_LIMIT_PCT
            for row in selected_buckets
        )
        numeric_accuracy_passed = (
            bool(comparable_rows)
            and not case_violations
            and not accuracy_bucket_violations)
        scopes.append({
            "result_scope": result_scope,
            "row_count": len(rows),
            "comparable_row_count": len(comparable_rows),
            "not_comparable_row_count": sum(
                _comparison_status(row) == "not_comparable"
                for row in rows),
            "unobserved_row_count": sum(
                _comparison_status(row) == "unobserved"
                for row in rows),
            "bucket_count": len(selected_buckets),
            "mape_pct": None if metrics is None else metrics["mape_pct"],
            "p99_ape_pct": (
                None if metrics is None else metrics["p99_ape_pct"]),
            "max_ape_pct": (
                None if metrics is None else metrics["max_ape_pct"]),
            "mean_signed_error_pct": (
                None if metrics is None
                else metrics["mean_signed_error_pct"]),
            "diagnostic_mape_pct": diagnostic_metrics["mape_pct"],
            "diagnostic_p99_ape_pct": diagnostic_metrics["p99_ape_pct"],
            "diagnostic_max_ape_pct": diagnostic_metrics["max_ape_pct"],
            "diagnostic_mean_signed_error_pct": diagnostic_metrics[
                "mean_signed_error_pct"],
            "case_violation_count": len(case_violations),
            "diagnostic_case_violation_count": (
                diagnostic_case_violation_count),
            "diagnostic_bucket_mape_violation_count": (
                diagnostic_bucket_mape_violation_count),
            "diagnostic_bucket_max_ape_violation_count": (
                diagnostic_bucket_max_ape_violation_count),
            "comparison_row_violation_count": len(
                comparison_row_violations),
            "comparison_bucket_violation_count": len(
                comparison_bucket_violations),
            "accuracy_bucket_violation_count": len(
                accuracy_bucket_violations),
            "support_bucket_violation_count": len(
                support_bucket_violations),
            "goal_bucket_violation_count": len(goal_bucket_violations),
            "comparison_coverage_gate_passed": (
                comparison_coverage_passed),
            "numeric_accuracy_gate_passed": numeric_accuracy_passed,
            "accuracy_gate_passed": (
                comparison_coverage_passed and numeric_accuracy_passed),
            "model_closure_gate_passed": (
                comparison_coverage_passed
                and numeric_accuracy_passed
                and not goal_bucket_violations),
        })
    all_case_violations.sort(key=lambda row: (
        -float(row["absolute_error_pct"]), row["result_scope"],
        row["hardware_id"], row["case_id"],
        -1 if row["kernel_ordinal"] is None else int(row["kernel_ordinal"]),
    ))
    all_accuracy_bucket_violations.sort(key=lambda row: (
        -float(row["mape_pct"]), row["result_scope"],
        row["physical_bucket_id"],
    ))
    all_comparison_bucket_violations.sort(key=lambda row: (
        float(row["comparison_coverage_pct"]), row["result_scope"],
        row["physical_bucket_id"],
    ))
    all_comparison_row_violations.sort(key=lambda row: (
        row["result_scope"], row["hardware_id"], row["family"],
        row["case_id"],
        -1 if row["kernel_ordinal"] is None else int(row["kernel_ordinal"]),
    ))
    all_support_bucket_violations.sort(key=lambda row: (
        float(row["complete_support_pct"]), row["result_scope"],
        row["physical_bucket_id"],
    ))
    all_goal_bucket_violations.sort(key=lambda row: (
        -float(row["diagnostic_mape_pct"]), row["result_scope"],
        row["physical_bucket_id"],
    ))
    return {
        "schema_version": 1,
        "bucket_schema": BUCKET_SCHEMA_VERSION,
        "thresholds": {
            "bucket_mape_pct_max_inclusive": BUCKET_MAPE_LIMIT_PCT,
            "case_ape_pct_max_inclusive": CASE_APE_LIMIT_PCT,
            "complete_support_pct_required": 100.0,
        },
        "mape_definition": (
            "arithmetic mean of per-row absolute percentage error; signed "
            "errors are never allowed to cancel"),
        "diagnostic_metric_policy": (
            "diagnostic_* fields retain arithmetic discrepancies for triage; "
            "only rows with comparison_status=comparable enter the accuracy "
            "denominator"),
        "scopes": scopes,
        "case_violations": all_case_violations,
        "comparison_row_violations": all_comparison_row_violations,
        "comparison_bucket_violations": all_comparison_bucket_violations,
        "accuracy_bucket_violations": all_accuracy_bucket_violations,
        "support_bucket_violations": all_support_bucket_violations,
        "goal_bucket_violations": all_goal_bucket_violations,
        "accuracy_gate_passed": all(
            row["accuracy_gate_passed"] for row in scopes),
        "model_closure_gate_passed": all(
            row["model_closure_gate_passed"] for row in scopes),
    }


def _physical_bucket_definitions() -> Mapping[str, Any]:
    return {
        "schema_version": BUCKET_SCHEMA_VERSION,
        "classification_boundary": (
            "workload and observed discrete launch facts only; prediction, "
            "duration, counters, and errors are forbidden inputs"),
        "result_scopes": {
            "real_kernel": (
                "one profiler kernel invocation with its own duration"),
            "aggregate_task": (
                "one canonical dataset task; may contain multiple kernels"),
        },
        "task_execution_signature": (
            "ordered full static kernel launch classes; duplicates are "
            f"preserved; task aggregation is {TASK_AGGREGATION_CONTRACT}"),
        "task_composition_context": (
            "task family is report context, not a bucket dimension; "
            "FlashAttention includes layout at both result scopes"),
        "kernel_launch_class": {
            "dimensions": [
                "hardware_id", "kernel_kind", "variant", "dtype",
                "execution_variant", "algorithm", "grid_rank", "block",
                "cluster", "source_program_type", "tiling",
                "discrete_controls",
            ],
            "excluded_shape_dependent_fields": [
                "grid extents", "grid_size", "padded_rows", "operation counts",
                "traffic byte counts", "worker operation counts",
            ],
            "excluded_observation_identity_fields": [
                "runtime_symbol",
            ],
            "discrete_controls": list(PHYSICAL_BUCKET_DISCRETE_CONTROLS),
        },
        "family_semantics": {
            "GEMM": [],
            "FlashAttention": ["attention_layout"],
            "FusedMoE": [],
            "Vector": [],
        },
        "small_sample_policy": (
            "never merge buckets using observed errors; report row_count and "
            "retain singleton buckets as evidence_low_n"),
        "diagnostic_only": {
            "derived_launch_rounds": (
                "ceil(grid_size/execution_units); not a profiler-observed "
                "wave counter and never a primary bucket dimension"),
        },
    }


def _component_ratio_summary(
        rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    positive = [
        row for row in rows
        if row["predicted"] > 0 and row["observed"] > 0
    ]
    zero_mismatch = sum(
        (row["predicted"] == 0) != (row["observed"] == 0)
        for row in rows)
    if positive:
        ratios = np.asarray([
            row["predicted"] / row["observed"] for row in positive],
            dtype=float,
        )
        absolute = np.abs(ratios - 1.0) * 100.0
        median_ratio = float(np.median(ratios))
        p10_ratio = float(np.percentile(ratios, 10))
        p90_ratio = float(np.percentile(ratios, 90))
        mape_pct = float(np.mean(absolute))
    else:
        median_ratio = None
        p10_ratio = None
        p90_ratio = None
        mape_pct = None
    return {
        "positive_pair_count": len(positive),
        "zero_mismatch_count": zero_mismatch,
        "median_predicted_over_observed": median_ratio,
        "p10_predicted_over_observed": p10_ratio,
        "p90_predicted_over_observed": p90_ratio,
        "mape_pct": mape_pct,
    }


def _comparison_summary(
        comparison_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    groups = {}
    for row in comparison_rows:
        key = (row["hardware_id"], row["family"], row["component"], row["unit"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    result = []
    for key in sorted(groups):
        hardware_id, family, component, unit = key
        values = groups[key]
        statuses = [_comparison_status(row) for row in values]
        comparable = [
            row for row in values
            if _comparison_status(row) == "comparable"
        ]
        metrics = _component_ratio_summary(comparable)
        diagnostic_metrics = _component_ratio_summary(values)
        result.append({
            "hardware_id": hardware_id,
            "family": family,
            "component": component,
            "unit": unit,
            "row_count": len(values),
            "comparable_count": sum(
                status == "comparable" for status in statuses),
            "not_comparable_count": sum(
                status == "not_comparable" for status in statuses),
            "unobserved_count": sum(
                status == "unobserved" for status in statuses),
            **metrics,
            "diagnostic_positive_pair_count": diagnostic_metrics[
                "positive_pair_count"],
            "diagnostic_zero_mismatch_count": diagnostic_metrics[
                "zero_mismatch_count"],
            "diagnostic_median_predicted_over_observed": diagnostic_metrics[
                "median_predicted_over_observed"],
            "diagnostic_p10_predicted_over_observed": diagnostic_metrics[
                "p10_predicted_over_observed"],
            "diagnostic_p90_predicted_over_observed": diagnostic_metrics[
                "p90_predicted_over_observed"],
            "diagnostic_mape_pct": diagnostic_metrics["mape_pct"],
        })
    return result


def _traffic_outlier_rows(
        comparison_rows: Sequence[Mapping[str, Any]],
        relative_error_threshold: float = 0.5) -> list[Mapping[str, Any]]:
    """Return one self-contained row per severe directional traffic pair.

    A row is severe when its observed-relative error is at least the requested
    threshold.  Zero/non-zero mismatches are retained explicitly: predicted
    zero versus observed positive has 100% error, while observed zero versus
    predicted positive has no finite observed-relative percentage and is
    labelled as an infinite overprediction.  Dropping either case would hide
    exactly the missing-write and writeback-boundary defects this report is
    intended to expose.
    """

    if not 0.0 < relative_error_threshold < 1.0:
        raise ValueError("traffic outlier threshold must be within (0, 1)")
    traffic_components = frozenset({
        "demand:l2_cache",
        "demand:dram",
        *DIRECTIONAL_TRAFFIC_COMPONENTS,
    })
    result = []
    for row in comparison_rows:
        component = row["component"]
        if component not in traffic_components:
            continue
        predicted = float(row["predicted"])
        observed = float(row["observed"])
        if (not math.isfinite(predicted) or predicted < 0
                or not math.isfinite(observed) or observed < 0):
            raise ValueError(
                "traffic comparison values must be finite and non-negative: "
                f"component={component!r}, predicted={predicted!r}, "
                f"observed={observed!r}")
        if observed > 0:
            ratio = predicted / observed
            absolute_error_pct = abs(predicted - observed) / observed * 100.0
            zero_mismatch = (
                "predicted_zero_observed_positive"
                if predicted == 0 else None)
            severe = absolute_error_pct >= relative_error_threshold * 100.0
        elif predicted > 0:
            ratio = None
            absolute_error_pct = None
            zero_mismatch = "observed_zero_predicted_positive"
            severe = True
        else:
            continue
        if not severe:
            continue
        result.append({
            "hardware_id": row["hardware_id"],
            "family": row["family"],
            "case_id": row["case_id"],
            "kernel_ordinal": row["kernel_ordinal"],
            "variant": row["variant"],
            "kernel_name": row["kernel_name"],
            "actual_ns": row["actual_ns"],
            "timing_source": row["timing_source"],
            "component": component,
            "predicted_bytes": predicted,
            "observed_bytes": observed,
            "predicted_over_observed": ratio,
            "absolute_error_pct": absolute_error_pct,
            "zero_mismatch": zero_mismatch,
            "severity_reason": (
                "zero_mismatch" if zero_mismatch is not None
                else "observed_relative_error_threshold_met"),
            "relative_error_threshold_pct": (
                relative_error_threshold * 100.0),
            "observation_metric": row["metric"],
            "provenance_kind": row["provenance_kind"],
            "provider": row["provider"],
            "comparison_status": _comparison_status(row),
            "comparison_reason": row["comparison_reason"],
        })
    result.sort(key=lambda row: (
        row["actual_ns"] is None,
        -float(row["actual_ns"] or 0.0),
        row["hardware_id"],
        row["family"],
        row["case_id"],
        int(row["kernel_ordinal"]),
        row["component"],
    ))
    return result


def _ascend_moe_tile_replay_rows(
        documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Compare production GroupedMatmul paths with matching interfaces."""

    rows = []
    for document in documents:
        if document["hardware_id"] != "ascend_910b":
            continue
        for task in document["tasks"]:
            if task["family"] != "FusedMoE":
                continue
            for kernel in task["kernels"]:
                mechanism = kernel["mechanisms"]
                variant = mechanism["variant"]
                if variant not in ("w1", "w2"):
                    continue
                launch = mechanism["observed_launch"]
                tiling = launch["tiling"]
                if tiling is None:
                    raise ValueError(
                        "Ascend GroupedMatmul replay check requires tiling")
                tile_shape = tiling["tile_shape"]
                tile_m = int(tile_shape["m"])
                tile_n = int(tile_shape["n"])
                tile_k = int(tile_shape["k"])
                if min(tile_m, tile_n, tile_k) <= 0:
                    raise ValueError(
                        "Ascend GroupedMatmul replay tiles must be positive")
                paths = mechanism["analytical_input"]["memory_paths"]
                if not isinstance(paths, list):
                    raise TypeError(
                        "Ascend GroupedMatmul memory paths must be a list")
                paths_by_name = {path["name"]: path for path in paths}
                if (len(paths_by_name) != len(paths)
                        or set(paths_by_name)
                        != {
                            "mte2_read", "operand_dram_read",
                            "aic_output_l2_write",
                        }):
                    raise ValueError(
                        "Ascend GroupedMatmul requires one MTE2 L2-read path, "
                        "one operand DRAM-read path, and one AIC output "
                        "L2-write path")
                read_path = paths_by_name["mte2_read"]
                dram_read_path = paths_by_name["operand_dram_read"]
                output_path = paths_by_name["aic_output_l2_write"]
                if (read_path["access_kind"] != "read"
                        or dram_read_path["access_kind"] != "read"
                        or output_path["access_kind"] != "write"):
                    raise ValueError(
                        "Ascend GroupedMatmul path access kinds are invalid")
                read_nodes = {
                    node["resource_role"]: float(node["bytes"])
                    for node in read_path["nodes"]
                }
                dram_read_nodes = {
                    node["resource_role"]: float(node["bytes"])
                    for node in dram_read_path["nodes"]
                }
                output_nodes = {
                    node["resource_role"]: float(node["bytes"])
                    for node in output_path["nodes"]
                }
                if set(read_nodes) != {"l2_cache"}:
                    raise ValueError(
                        "Ascend GroupedMatmul MTE2 read nodes are invalid")
                if set(dram_read_nodes) != {"dram"}:
                    raise ValueError(
                        "Ascend GroupedMatmul operand DRAM-read nodes are "
                        "invalid")
                if set(output_nodes) != {"l2_cache", "dram"}:
                    raise ValueError(
                        "Ascend GroupedMatmul output-write nodes are invalid")
                if (dram_read_nodes["dram"] > read_nodes["l2_cache"]
                        or output_nodes["dram"] > output_nodes["l2_cache"]):
                    raise ValueError(
                        "Ascend GroupedMatmul DRAM continuation cannot exceed "
                        "its directional L2 interface")
                predicted_read_bytes = read_nodes["l2_cache"]
                predicted_output_bytes = output_nodes["l2_cache"]
                observed_read_bytes = float(
                    mechanism["observation"]["native"]["gm_to_l1_bytes"])
                observed_output_bytes = float(
                    mechanism["observation"]["l2_interface_write_bytes"])
                if (predicted_read_bytes <= 0 or observed_read_bytes <= 0
                        or predicted_output_bytes <= 0
                        or observed_output_bytes <= 0):
                    raise ValueError(
                        "Ascend GroupedMatmul interface bytes must be positive")
                native = mechanism["observation"]["native"]
                if native["ub_to_gm_kb"] != "NA":
                    raise ValueError(
                        "Ascend GroupedMatmul UB-to-GM must remain unavailable")
                rows.append({
                    "hardware_id": document["hardware_id"],
                    "family": task["family"],
                    "case_id": task["case_id"],
                    "kernel_ordinal": mechanism["kernel_ordinal"],
                    "variant": variant,
                    "actual_ns": mechanism["timing_target"]["duration_ns"],
                    "timing_source": mechanism["timing_target"]["source"],
                    **_canonical_capture_comparison_evidence(
                        document["hardware_id"]),
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "tile_k": tile_k,
                    "predicted_mte2_l2_read_bytes": predicted_read_bytes,
                    "observed_gm_to_l1_read_bytes": observed_read_bytes,
                    "read_predicted_over_observed": (
                        predicted_read_bytes / observed_read_bytes),
                    "read_absolute_error_pct": (
                        abs(predicted_read_bytes - observed_read_bytes)
                        / observed_read_bytes * 100.0),
                    "predicted_aic_output_l2_write_bytes": (
                        predicted_output_bytes),
                    "observed_aic_main_memory_write_bytes": (
                        observed_output_bytes),
                    "output_predicted_over_observed": (
                        predicted_output_bytes / observed_output_bytes),
                    "output_absolute_error_pct": (
                        abs(predicted_output_bytes - observed_output_bytes)
                        / observed_output_bytes * 100.0),
                    "modeled_dram_read_bytes": dram_read_nodes["dram"],
                    "observed_dram_write_bytes": float(
                        mechanism["observation"]["dram_write_bytes"]),
                    "dram_residency_status": "evidence_pending",
                })
    rows.sort(key=lambda row: (
        row["actual_ns"] is None,
        -float(row["actual_ns"] or 0.0),
        row["case_id"],
        int(row["kernel_ordinal"]),
    ))
    return rows


def _c500_silu_mechanism_rows(
        documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep C500 SiLU demand checks separate from replay timing counters."""

    rows = []
    for document in documents:
        if document["hardware_id"] != "metax_c500":
            continue
        for task in document["tasks"]:
            if task["family"] != "Vector":
                continue
            for kernel in task["kernels"]:
                mechanism = kernel["mechanisms"]
                if mechanism["variant"] != "silu":
                    continue
                if mechanism["kernel_kind"] != "VECTOR":
                    raise ValueError("C500 SiLU mechanism row must be VECTOR")
                analytical = mechanism["analytical_input"]
                observation = mechanism["observation"]
                native = observation["native"]
                if native["backend"] != "metax_mcprofiler":
                    raise ValueError(
                        "C500 SiLU mechanism row requires mcProfiler counters")
                timing_target = mechanism["timing_target"]
                if "availability" not in timing_target:
                    timing_availability = "observed"
                else:
                    timing_availability = timing_target["availability"]
                if timing_availability == "observed":
                    result_scope = "real_kernel"
                    production_duration_ns = float(
                        timing_target["duration_ns"])
                    predicted_duration_ns = float(
                        mechanism["forward"]["total_ns"])
                    timing_source = timing_target["source"]
                elif timing_availability == "unobserved":
                    if (timing_target["duration_ns"] is not None
                            or timing_target["source"] is not None
                            or timing_target["unavailable_reason"]
                            != "canonical_target_is_op_level"):
                        raise ValueError(
                            "C500 SiLU unobserved kernel target is malformed")
                    if task["metadata"]["actual_granularity"] != "op":
                        raise ValueError(
                            "C500 SiLU aggregate timing must be OP-level")
                    result_scope = "aggregate_task"
                    predicted_duration_ns, production_duration_ns = (
                        float(value) for value in task["forward_pair"])
                    timing_source = task["metadata"]["actual_source"]
                else:
                    raise ValueError(
                        "unknown C500 SiLU timing availability: "
                        f"{timing_availability!r}")
                if (timing_source != "device_event_op_median"
                        or native["duration_source"] != "mcTracer"):
                    raise ValueError(
                        "C500 SiLU canonical duration must come from the "
                        "device-event target while replay counters retain "
                        "their mcTracer duration provenance")
                shape = mechanism["shape"]
                tokens = int(shape["tok"])
                intermediate = int(shape["intermediate"])
                predicted_instructions = float(
                    analytical["demand_per_resource"]["warp_issue"])
                observed_instructions = float(native["total_instructions"])
                predicted_global_bytes = float(
                    analytical["demand_per_resource"]["dram"])
                observed_global_bytes = float(
                    native["global_memory_access_bytes"])
                instrumented_duration_ns = float(
                    native["instrumented_during_ns"])
                if native["core_clock_ghz"] is None:
                    raise ValueError(
                        "C500 SiLU mechanism row requires mcProfiler "
                        "core_clk")
                instrumented_core_clock_ghz = float(
                    native["core_clock_ghz"])
                instrumented_active_cycles = float(
                    observation["active_cycles"])
                instrumented_cycles = (
                    instrumented_duration_ns
                    * instrumented_core_clock_ghz)
                positive_values = {
                    "production_duration_ns": production_duration_ns,
                    "predicted_duration_ns": predicted_duration_ns,
                    "observed_total_instructions": observed_instructions,
                    "predicted_global_bytes": predicted_global_bytes,
                    "observed_global_bytes": observed_global_bytes,
                    "instrumented_duration_ns": instrumented_duration_ns,
                    "instrumented_core_clock_ghz": (
                        instrumented_core_clock_ghz),
                    "instrumented_ap_active_cycles": instrumented_active_cycles,
                    "instrumented_cycles": instrumented_cycles,
                    "waves": float(native["waves"]),
                    "instrumented_average_wave_life_cycles": float(
                        native["average_wave_life_cycles"]),
                }
                invalid = {
                    name: value for name, value in positive_values.items()
                    if not math.isfinite(value) or value <= 0
                }
                if invalid:
                    raise ValueError(
                        "C500 SiLU mechanism row has invalid positive values: "
                        f"{invalid}")
                if (not math.isfinite(predicted_instructions)
                        or predicted_instructions < 0):
                    raise ValueError(
                        "C500 SiLU predicted instruction lower bound must be "
                        "finite and non-negative")
                if predicted_instructions > 0:
                    instruction_ratio = (
                        predicted_instructions / observed_instructions)
                    instruction_status = "comparable"
                    instruction_reason = ""
                else:
                    instruction_ratio = None
                    instruction_status = "unavailable"
                    instruction_reason = (
                        "Forward publishes no static instruction lower bound "
                        "for this C500 SiLU source revision")
                non_negative_values = {
                    "instrumented_dnoc_read_latency_cycles": float(
                        native["dnoc_read_average_latency_cycles"]),
                    "instrumented_vls_pipeline_stall_cycles": float(
                        native["vls_pipeline_stall_cycles"]),
                    "instrumented_vls_wdata_stall_cycles": float(
                        native["vls_wdata_stall_cycles"]),
                }
                invalid_non_negative = {
                    name: value
                    for name, value in non_negative_values.items()
                    if not math.isfinite(value) or value < 0
                }
                if invalid_non_negative:
                    raise ValueError(
                        "C500 SiLU mechanism row has invalid non-negative "
                        f"values: {invalid_non_negative}")
                instrumented_mean_active_aps = (
                    instrumented_active_cycles / instrumented_cycles)
                active_units = int(analytical["active_units"])
                if instrumented_mean_active_aps > active_units * 1.01:
                    raise ValueError(
                        "C500 SiLU replay active-AP count exceeds the launch "
                        "capacity")
                rows.append({
                    "result_scope": result_scope,
                    "case_id": task["case_id"],
                    "tokens": tokens,
                    "intermediate": intermediate,
                    "production_duration_ns": production_duration_ns,
                    "timing_source": timing_source,
                    "predicted_duration_ns": predicted_duration_ns,
                    "predicted_over_production": (
                        predicted_duration_ns / production_duration_ns),
                    "absolute_error_pct": (
                        abs(predicted_duration_ns - production_duration_ns)
                        / production_duration_ns * 100.0),
                    "grid_size": int(analytical["grid_size"]),
                    "waves": positive_values["waves"],
                    "predicted_instruction_lower_bound": (
                        predicted_instructions),
                    "observed_total_instructions": observed_instructions,
                    "instruction_predicted_over_observed": instruction_ratio,
                    "instruction_comparison_status": instruction_status,
                    "instruction_comparison_reason": instruction_reason,
                    "predicted_global_bytes": predicted_global_bytes,
                    "observed_global_bytes": observed_global_bytes,
                    "traffic_predicted_over_observed": (
                        predicted_global_bytes / observed_global_bytes),
                    "instrumented_duration_ns": instrumented_duration_ns,
                    "instrumented_core_clock_ghz": (
                        instrumented_core_clock_ghz),
                    "instrumented_ap_active_cycles": (
                        instrumented_active_cycles),
                    "instrumented_mean_active_aps": (
                        instrumented_mean_active_aps),
                    "instrumented_instruction_per_active_cycle": (
                        observed_instructions / instrumented_active_cycles),
                    "instrumented_average_wave_life_cycles": float(
                        positive_values[
                            "instrumented_average_wave_life_cycles"]),
                    "instrumented_dnoc_read_latency_cycles": float(
                        non_negative_values[
                            "instrumented_dnoc_read_latency_cycles"]),
                    "instrumented_vls_pipeline_stall_cycles": float(
                        non_negative_values[
                            "instrumented_vls_pipeline_stall_cycles"]),
                    "instrumented_vls_wdata_stall_cycles": float(
                        non_negative_values[
                            "instrumented_vls_wdata_stall_cycles"]),
                    "timing_scope_note": (
                        "canonical duration and Forward prediction use the "
                        f"{result_scope} boundary; AP/wave/DNoC/stall counters "
                        "and their mcTracer duration are a separate mcProfiler "
                        "replay and are never divided by the canonical "
                        "duration"),
                })
    rows.sort(key=lambda row: (
        -float(row["production_duration_ns"]), row["case_id"]))
    return rows


def _c500_silu_mechanism_summary(
        rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Summarize only directly comparable demand and same-replay evidence."""

    if not rows:
        raise ValueError("C500 SiLU mechanism summary requires rows")
    duration_apes = np.asarray([
        float(row["absolute_error_pct"]) for row in rows])
    instruction_rows = [
        row for row in rows
        if row["instruction_comparison_status"] == "comparable"
    ]
    instruction_apes = np.asarray([
        abs(float(row["instruction_predicted_over_observed"]) - 1.0) * 100.0
        for row in instruction_rows
    ])
    traffic_apes = np.asarray([
        abs(float(row["traffic_predicted_over_observed"]) - 1.0) * 100.0
        for row in rows
    ])
    largest_absolute_delta = max(
        rows,
        key=lambda row: abs(
            float(row["predicted_duration_ns"])
            - float(row["production_duration_ns"])),
    )
    return {
        "case_count": len(rows),
        "underpredicted_count": sum(
            float(row["predicted_duration_ns"])
            < float(row["production_duration_ns"])
            for row in rows),
        "case_gate_passed_count": int(np.sum(duration_apes <= CASE_APE_LIMIT_PCT)),
        "duration_mape_pct": float(np.mean(duration_apes)),
        "duration_max_ape_pct": float(np.max(duration_apes)),
        "instruction_comparison_status": (
            "comparable" if instruction_rows else "unavailable"),
        "instruction_comparable_count": len(instruction_rows),
        "instruction_exact_count": sum(
            float(row["predicted_instruction_lower_bound"])
            == float(row["observed_total_instructions"])
            for row in instruction_rows),
        "instruction_mape_pct": (
            float(np.mean(instruction_apes)) if instruction_rows else None),
        "instruction_max_ape_pct": (
            float(np.max(instruction_apes)) if instruction_rows else None),
        "traffic_mape_pct": float(np.mean(traffic_apes)),
        "traffic_max_ape_pct": float(np.max(traffic_apes)),
        "replay_mean_active_aps_min": min(
            float(row["instrumented_mean_active_aps"]) for row in rows),
        "replay_mean_active_aps_max": max(
            float(row["instrumented_mean_active_aps"]) for row in rows),
        "replay_core_clock_ghz_min": min(
            float(row["instrumented_core_clock_ghz"]) for row in rows),
        "replay_core_clock_ghz_max": max(
            float(row["instrumented_core_clock_ghz"]) for row in rows),
        "replay_wave_life_cycles_min": min(
            float(row["instrumented_average_wave_life_cycles"])
            for row in rows),
        "replay_wave_life_cycles_max": max(
            float(row["instrumented_average_wave_life_cycles"])
            for row in rows),
        "replay_dnoc_latency_cycles_min": min(
            float(row["instrumented_dnoc_read_latency_cycles"])
            for row in rows),
        "replay_dnoc_latency_cycles_max": max(
            float(row["instrumented_dnoc_read_latency_cycles"])
            for row in rows),
        "largest_absolute_delta_case_id": largest_absolute_delta["case_id"],
        "largest_absolute_delta_production_ns": float(
            largest_absolute_delta["production_duration_ns"]),
        "largest_absolute_delta_predicted_ns": float(
            largest_absolute_delta["predicted_duration_ns"]),
        "largest_absolute_delta_ape_pct": float(
            largest_absolute_delta["absolute_error_pct"]),
    }


def _coverage_rows(
        documents: Sequence[Mapping[str, Any]],
        comparison_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    components = (
        *(f"demand:{role}" for role in DEMAND_ROLES),
        *DIRECTIONAL_TRAFFIC_COMPONENTS,
        *(f"service_time:{role}" for role in TIME_ROLES),
    )
    evidenced = {
        (hardware_id, family, component): 0
        for hardware_id, _ in HARDWARE
        for family in FAMILY_LABELS
        for component in components
    }
    positive_pairs = dict.fromkeys(evidenced, 0)
    for row in comparison_rows:
        key = (row["hardware_id"], row["family"], row["component"])
        if key not in evidenced:
            raise ValueError(f"unexpected evidence component: {key!r}")
        evidenced[key] += 1
        if row["predicted"] > 0 and row["observed"] > 0:
            positive_pairs[key] += 1
    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        for family in FAMILY_LABELS:
            kernels = [
                kernel
                for task in document["tasks"] if task["family"] == family
                for kernel in task["kernels"]
            ]
            if not kernels:
                raise RuntimeError(
                    f"{hardware_id} has no {family} kernels")
            for component in components:
                key = (hardware_id, family, component)
                rows.append({
                    "hardware_id": hardware_id,
                    "family": family,
                    "component": component,
                    "kernel_count": len(kernels),
                    "profiler_evidenced_count": evidenced[key],
                    "positive_pair_count": positive_pairs[key],
                })
    return rows


def _plot_final_latency(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(
        FAMILY_LABELS,
        ("#0072B2", "#009E73", "#D55E00", "#CC79A7"),
        strict=True,
    ))
    figure, axes = plt.subplots(
        3, 4, figsize=(16.8, 12.0), squeeze=False)
    for row_index, (hardware_id, title) in enumerate(HARDWARE):
        for column_index, family in enumerate(FAMILY_LABELS):
            axis = axes[row_index][column_index]
            selected = [
                row for row in task_rows
                if row["hardware_id"] == hardware_id
                and row["family"] == family
            ]
            if not selected:
                raise RuntimeError(
                    f"no final-latency rows for {hardware_id}/{family}")
            observed = np.asarray([
                row["observed_ns"] / 1000.0 for row in selected])
            predicted = np.asarray([
                row["predicted_ns"] / 1000.0 for row in selected])
            lower = 10 ** math.floor(math.log10(min(
                float(np.min(observed)), float(np.min(predicted)))))
            upper = 10 ** math.ceil(math.log10(max(
                float(np.max(observed)), float(np.max(predicted)))))
            guide = np.asarray([lower, upper])
            axis.fill_between(
                guide, guide * 0.8, guide * 1.2,
                color="#009E73", alpha=0.10)
            axis.plot(
                guide, guide * 0.6, color="#D55E00", linewidth=0.8,
                linestyle=":")
            axis.plot(
                guide, guide * 1.4, color="#D55E00", linewidth=0.8,
                linestyle=":")
            axis.plot(
                guide, guide, color="#444444", linestyle="--", linewidth=1)
            axis.scatter(
                observed, predicted, s=10, alpha=0.45,
                color=colors[family], edgecolors="none", rasterized=True)
            worst = max(
                selected,
                key=lambda row: (
                    float(row["absolute_duration_error_ns"]),
                    float(row["absolute_error_pct"]),
                    float(row["observed_ns"]),
                    row["case_id"],
                ),
            )
            worst_observed = worst["observed_ns"] / 1000.0
            worst_predicted = worst["predicted_ns"] / 1000.0
            midpoint = math.sqrt(lower * upper)
            axis.annotate(
                f"largest |Δt|: {_plot_case_id(worst['case_id'])}\n"
                f"APE {worst['absolute_error_pct']:.1f}%",
                (worst_observed, worst_predicted),
                xytext=(-5 if worst_observed > midpoint else 5, 8),
                textcoords="offset points",
                horizontalalignment=(
                    "right" if worst_observed > midpoint else "left"),
                fontsize=6.5,
                arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                            "color": "#666666"},
            )
            axis.set(
                xscale="log", yscale="log",
                xlim=(lower, upper), ylim=(lower, upper))
            axis.set_title(f"{title} · {family} · n={len(selected)}")
            axis.set_xlabel("Observed task duration (µs)")
            axis.set_ylabel("Predicted task duration (µs)")
            axis.grid(True, which="major", alpha=0.25)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · task-duration log-log overview only: "
        f"center exact, green ±{BUCKET_MAPE_LIMIT_PCT:.0f}%, "
        f"dotted ±{CASE_APE_LIMIT_PCT:.0f}%")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_duration_ratio(
        rows: Sequence[Mapping[str, Any]], path: Path,
        result_scope: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if result_scope == "real_kernel":
        observed_field = "actual_ns"
        scope_label = "kernel"
    elif result_scope == "aggregate_task":
        observed_field = "observed_ns"
        scope_label = "task"
    else:
        raise ValueError(f"unknown duration-ratio scope: {result_scope!r}")
    colors = dict(zip(
        FAMILY_LABELS,
        ("#0072B2", "#009E73", "#D55E00", "#CC79A7"),
        strict=True,
    ))
    ratio_upper = max(
        float(row["predicted_ns"]) / float(row[observed_field])
        for row in rows) * 1.06
    ratio_upper = max(ratio_upper, 1.5)
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), sharey=True)
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        hardware_rows = [
            row for row in rows if row["hardware_id"] == hardware_id]
        if not hardware_rows:
            axis.text(
                0.5, 0.5, "No per-kernel timing",
                ha="center", va="center", transform=axis.transAxes)
            axis.set_title(title)
            axis.set_axis_off()
            continue
        for family in FAMILY_LABELS:
            selected = [
                row for row in hardware_rows if row["family"] == family]
            ratios = np.asarray([
                row["predicted_ns"] / row[observed_field]
                for row in selected
            ])
            axis.scatter(
                [row[observed_field] / 1000.0 for row in selected],
                ratios,
                s=10, alpha=0.45, edgecolors="none",
                color=colors[family], label=family, rasterized=True,
            )
        axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.10)
        axis.axhline(1.0, color="#333333", linewidth=1.1, linestyle="--")
        axis.axhline(0.6, color="#D55E00", linewidth=0.9, linestyle=":")
        axis.axhline(1.4, color="#D55E00", linewidth=0.9, linestyle=":")
        axis.set_xscale("log")
        axis.set_ylim(0.0, ratio_upper)
        worst = max(
            hardware_rows,
            key=lambda row: (
                float(row["absolute_duration_error_ns"]),
                float(row["absolute_error_pct"]),
                float(row[observed_field]),
                row["case_id"],
            ),
        )
        worst_ratio = worst["predicted_ns"] / worst[observed_field]
        worst_x = worst[observed_field] / 1000.0
        ordinal = _kernel_ordinal_suffix(worst, result_scope)
        observed_values = [
            float(row[observed_field]) / 1000.0 for row in hardware_rows]
        horizontal_midpoint = math.sqrt(
            min(observed_values) * max(observed_values))
        place_left = worst_x > horizontal_midpoint
        place_below = worst_ratio >= 1.0
        axis.annotate(
            f"largest |Δt|: {_plot_case_id(worst['case_id'])}{ordinal}\n"
            f"{worst_ratio:.3g}×, APE {worst['absolute_error_pct']:.1f}%",
            (worst_x, worst_ratio),
            xytext=(-6 if place_left else 6,
                    -10 if place_below else 10),
            textcoords="offset points",
            horizontalalignment="right" if place_left else "left",
            verticalalignment="top" if place_below else "bottom",
            fontsize=6.5,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                        "color": "#666666"},
        )
        axis.set_title(title)
        axis.set_xlabel(f"Observed {scope_label} duration (µs)")
        axis.set_ylabel("Predicted ÷ observed duration")
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}×"))
        axis.grid(True, which="major", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=False)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · "
        f"{scope_label} duration ratio on a linear y-axis; "
        f"green ±{BUCKET_MAPE_LIMIT_PCT:.0f}%, "
        f"dotted ±{CASE_APE_LIMIT_PCT:.0f}%",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_case_impact_ranking(
        impact_rows: Sequence[Mapping[str, Any]], path: Path,
        rows_per_panel: int = 10) -> None:
    """Plot the largest row-level accuracy failures on a linear time axis."""

    if (isinstance(rows_per_panel, bool)
            or not isinstance(rows_per_panel, int)
            or rows_per_panel <= 0):
        raise ValueError("case-impact rows_per_panel must be positive")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(
        2, 3, figsize=(19.0, 13.0), squeeze=False)
    scopes = (
        ("real_kernel", "Real kernel invocation"),
        ("aggregate_task", "Aggregate task"),
    )
    colors = {"under": "#0072B2", "over": "#D55E00"}
    for row_index, (result_scope, scope_title) in enumerate(scopes):
        for column_index, (hardware_id, hardware_title) in enumerate(HARDWARE):
            axis = axes[row_index][column_index]
            selected = [
                row for row in impact_rows
                if row["result_scope"] == result_scope
                and row["hardware_id"] == hardware_id
            ][:rows_per_panel]
            if not selected:
                axis.text(
                    0.5, 0.5,
                    f"No APE > {CASE_APE_LIMIT_PCT:.0f}% rows",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
                continue
            selected = list(reversed(selected))
            positions = np.arange(len(selected))
            impact_ms = np.asarray([
                float(row["absolute_duration_error_ns"]) / 1.0e6
                for row in selected
            ])
            bar_colors = [
                colors["under"]
                if float(row["signed_error_pct"]) < 0
                else colors["over"]
                for row in selected
            ]
            axis.barh(positions, impact_ms, color=bar_colors, alpha=0.82)
            labels = []
            for row in selected:
                suffix = _kernel_ordinal_suffix(row, result_scope)
                labels.append(
                    f"#{row['impact_rank']} "
                    f"{_plot_case_id(row['case_id'], width=34)}{suffix}")
            axis.set_yticks(positions, labels, fontsize=6.2)
            upper = max(impact_ms) * 1.62
            if upper <= 0:
                raise ValueError("case-impact plot requires positive deltas")
            axis.set_xlim(0.0, upper)
            for position, value, row in zip(
                    positions, impact_ms, selected, strict=True):
                predicted_ms = float(row["predicted_ns"]) / 1.0e6
                measured_ms = float(row["measured_ns"]) / 1.0e6
                axis.text(
                    value + upper * 0.012,
                    position,
                    f"APE {float(row['absolute_error_pct']):.1f}%\n"
                    f"P/O {predicted_ms:.3g}/{measured_ms:.3g} ms",
                    verticalalignment="center",
                    fontsize=6.2,
                )
            axis.set_title(f"{hardware_title} · {scope_title}")
            axis.set_xlabel("|Predicted − measured| (ms)")
            axis.grid(True, axis="x", alpha=0.25)
    figure.legend(
        handles=(
            Patch(facecolor=colors["under"], label="Underprediction"),
            Patch(facecolor=colors["over"], label="Overprediction"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · largest APE > "
        f"{CASE_APE_LIMIT_PCT:.0f}% rows "
        "ranked by absolute time impact "
        "(linear axis)",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=1.6)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_bucket_accuracy(bucket_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bucket_families = (*FAMILY_LABELS, "mixed")
    colors = dict(zip(
        bucket_families,
        ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#777777"),
        strict=True,
    ))
    error_upper = max(
        CASE_APE_LIMIT_PCT * 1.25,
        max(
            max(
                float(row["diagnostic_mape_pct"]),
                float(row["diagnostic_max_ape_pct"]))
            for row in bucket_rows) * 1.08,
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(15.8, 8.4), sharey=True, squeeze=False)
    for row_index, result_scope in enumerate(
            ("real_kernel", "aggregate_task")):
        for column_index, (hardware_id, title) in enumerate(HARDWARE):
            axis = axes[row_index][column_index]
            selected = sorted(
                (
                    row for row in bucket_rows
                    if row["result_scope"] == result_scope
                    and row["hardware_id"] == hardware_id
                ),
                key=lambda row: (
                    row["diagnostic_mape_pct"],
                    row["physical_bucket_id"]),
            )
            if not selected:
                axis.text(
                    0.5, 0.5, "No per-kernel timing",
                    ha="center", va="center", transform=axis.transAxes)
                axis.set_title(title)
                axis.set_axis_off()
                continue
            for family in bucket_families:
                positions = [
                    index for index, row in enumerate(selected)
                    if row["family"] == family
                ]
                if not positions:
                    continue
                axis.scatter(
                    positions,
                    [
                        selected[index]["diagnostic_mape_pct"]
                        for index in positions
                    ],
                    s=14, alpha=0.75, edgecolors="none",
                    color=colors[family], label=family, rasterized=True,
                )
            axis.scatter(
                range(len(selected)),
                [row["diagnostic_max_ape_pct"] for row in selected],
                s=10, marker="x", linewidths=0.65, color="#333333",
                label="Worst case in bucket", rasterized=True,
            )
            passed = sum(
                row["diagnostic_mape_pct"] <= BUCKET_MAPE_LIMIT_PCT
                and row["diagnostic_max_ape_pct"] <= CASE_APE_LIMIT_PCT
                for row in selected)
            full_passed = sum(row["bucket_gate_passed"] for row in selected)
            axis.axhline(
                BUCKET_MAPE_LIMIT_PCT, color="#D55E00", linewidth=1.0,
                linestyle="--")
            axis.axhline(
                CASE_APE_LIMIT_PCT, color="#666666", linewidth=1.0,
                linestyle=":")
            axis.set_ylim(0.0, error_upper)
            worst = max(
                selected,
                key=lambda row: (
                    float(row["diagnostic_mape_pct"]),
                    float(row["diagnostic_max_ape_pct"]),
                    row["physical_bucket_id"],
                ),
            )
            ordinal = (
                "" if worst["diagnostic_worst_kernel_ordinal"] is None
                else f" k{worst['diagnostic_worst_kernel_ordinal']}")
            axis.text(
                0.02,
                0.98,
                "diagnostic max\n"
                f"{_plot_case_id(worst['diagnostic_worst_case_id'])}"
                f"{ordinal}\n"
                f"bucket {worst['diagnostic_mape_pct']:.1f}%, "
                f"case {worst['diagnostic_max_ape_pct']:.1f}%",
                transform=axis.transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=6.5,
            )
            axis.set_title(
                f"{title}\ndiagnostic thresholds {passed}/{len(selected)}; "
                f"full closure {full_passed}/{len(selected)}")
            axis.set_xlabel("Buckets sorted by MAPE")
            if result_scope == "real_kernel":
                scope_label = "Kernel bucket"
            elif result_scope == "aggregate_task":
                scope_label = "Task bucket"
            else:
                raise AssertionError(
                    f"unhandled result scope: {result_scope!r}")
            axis.set_ylabel(f"{scope_label} error (%)")
            axis.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=False)
    figure.suptitle(
        "Provenance-unverified physical-bucket diagnostic; circles are MAPE, "
        f"x marks are worst cases; dashed {BUCKET_MAPE_LIMIT_PCT:.0f}%, "
        f"dotted {CASE_APE_LIMIT_PCT:.0f}%",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_before_after_family_accuracy(summary_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = [row for row in summary_rows if row["family"] != "ALL"]
    upper = max(
        CASE_APE_LIMIT_PCT * 1.2,
        max(
            max(
                float(row["before_mape_pct"]),
                float(row["after_mape_pct"]))
            for row in selected) * 1.08,
    )
    figure, axes = plt.subplots(1, 2, figsize=(15.8, 7.0), sharex=True)
    for axis, result_scope in zip(
            axes, ("real_kernel", "aggregate_task"), strict=True):
        values = [
            row for hardware_id, _ in HARDWARE
            for family in FAMILY_LABELS
            for row in selected
            if row["result_scope"] == result_scope
            and row["hardware_id"] == hardware_id
            and row["family"] == family
        ]
        if (result_scope == "aggregate_task"
                and len(values) != len(HARDWARE) * len(FAMILY_LABELS)):
            raise ValueError(
                f"before/after family plot is incomplete for {result_scope}")
        if not values:
            if result_scope != "real_kernel":
                raise AssertionError(
                    f"unhandled empty result scope: {result_scope!r}")
            axis.text(
                0.5,
                0.5,
                "No canonical real-kernel timings",
                transform=axis.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )
            axis.set_yticks([])
            axis.set_xlabel("Mean absolute percentage error (%)")
            axis.set_title(_result_scope_title(result_scope))
            axis.grid(True, axis="x", alpha=0.25)
            continue
        labels = [
            f"{next(title for identifier, title in HARDWARE if identifier == row['hardware_id'])} · {row['family']}"
            for row in values
        ]
        positions = np.arange(len(values))
        before = np.asarray(
            [float(row["before_mape_pct"]) for row in values])
        after = np.asarray(
            [float(row["after_mape_pct"]) for row in values])
        for position, before_value, after_value in zip(
                positions, before, after, strict=True):
            axis.plot(
                (before_value, after_value),
                (position, position),
                color="#777777",
                linewidth=1.0,
                zorder=1,
            )
        axis.scatter(
            before, positions, facecolors="none", edgecolors="#0072B2",
            linewidths=1.0, s=34, zorder=2)
        axis.scatter(
            after, positions, color="#D55E00", marker="s", s=28, zorder=3)
        axis.axvline(
            BUCKET_MAPE_LIMIT_PCT, color="#009E73", linestyle="--",
            linewidth=1.1)
        axis.set_xlim(0.0, upper)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Mean absolute percentage error (%)")
        axis.set_title(_result_scope_title(result_scope))
        axis.grid(True, axis="x", alpha=0.25)
    figure.legend(
        handles=(
            Line2D(
                (), (), marker="o", markerfacecolor="none",
                markeredgecolor="#0072B2", linestyle="none",
                label="Before"),
            Line2D(
                (), (), marker="s", color="#D55E00", linestyle="none",
                label="After"),
            Line2D(
                (), (), color="#009E73", linestyle="--",
                label=(
                    f"{BUCKET_MAPE_LIMIT_PCT:.0f}% bucket-mean target")),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · same rows before and after",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_before_after_bucket_accuracy(bucket_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {
        "nvidia_a800": "#0072B2",
        "metax_c500": "#D55E00",
        "ascend_910b": "#009E73",
    }
    titles = dict(HARDWARE)
    upper = max(
        CASE_APE_LIMIT_PCT * 1.2,
        max(
            max(
                float(row["before_mape_pct"]),
                float(row["after_mape_pct"]))
            for row in bucket_rows) * 1.08,
    )
    figure, axes = plt.subplots(1, 2, figsize=(14.6, 6.5), sharex=True, sharey=True)
    for axis, result_scope in zip(
            axes, ("real_kernel", "aggregate_task"), strict=True):
        selected = [
            row for row in bucket_rows
            if row["result_scope"] == result_scope]
        if not selected:
            raise ValueError(
                f"before/after bucket plot has no {result_scope} rows")
        for hardware_id, _ in HARDWARE:
            values = [
                row for row in selected
                if row["hardware_id"] == hardware_id]
            axis.scatter(
                [float(row["before_mape_pct"]) for row in values],
                [float(row["after_mape_pct"]) for row in values],
                color=colors[hardware_id],
                s=20,
                alpha=0.72,
                edgecolors="none",
                label=titles[hardware_id],
                rasterized=True,
            )
        max_violations = [
            row for row in selected
            if not row["after_diagnostic_max_ape_gate_passed"]]
        axis.scatter(
            [float(row["before_mape_pct"]) for row in max_violations],
            [float(row["after_mape_pct"]) for row in max_violations],
            marker="x", color="#333333", s=19, linewidths=0.65,
            label=f"Contains APE > {CASE_APE_LIMIT_PCT:.0f}%",
            rasterized=True)
        axis.plot(
            (0.0, upper), (0.0, upper), color="#666666", linewidth=1.0)
        axis.axvline(
            BUCKET_MAPE_LIMIT_PCT, color="#CC79A7", linestyle="--",
            linewidth=1.0)
        axis.axhline(
            BUCKET_MAPE_LIMIT_PCT, color="#CC79A7", linestyle="--",
            linewidth=1.0)
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)
        mape_before = sum(
            row["before_diagnostic_mape_gate_passed"] for row in selected)
        mape_after = sum(
            row["after_diagnostic_mape_gate_passed"] for row in selected)
        full_before = sum(
            row["before_diagnostic_accuracy_gate_passed"]
            for row in selected)
        full_after = sum(
            row["after_diagnostic_accuracy_gate_passed"]
            for row in selected)
        axis.set_title(
            _result_scope_title(result_scope)
            + " buckets"
            + f"\nMAPE pass {mape_before}→{mape_after}/{len(selected)}; "
            + f"both gates {full_before}→{full_after}")
        axis.set_xlabel("Before: bucket MAPE (%)")
        axis.set_ylabel("After: bucket MAPE (%)")
        axis.grid(True, alpha=0.2)
    handles = [
        Line2D(
            (), (), marker="o", color=colors[hardware_id],
            linestyle="none", label=title)
        for hardware_id, title in HARDWARE
    ]
    handles.extend((
        Line2D(
            (), (), marker="x", color="#333333", linestyle="none",
            label=("Bucket contains a case above "
                   f"{CASE_APE_LIMIT_PCT:.0f}%")),
        Line2D(
            (), (), color="#CC79A7", linestyle="--",
            label=f"{BUCKET_MAPE_LIMIT_PCT:.0f}% bucket-mean target"),
    ))
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · each point is one frozen physical bucket; lower is better",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_ape_cdf(
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(
        (hardware_id for hardware_id, _ in HARDWARE),
        ("#0072B2", "#D55E00", "#009E73"),
        strict=True,
    ))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    error_upper = max(
        CASE_APE_LIMIT_PCT * 1.25,
        max(float(row["absolute_error_pct"]) for row in kernel_rows) * 1.04,
        max(float(row["absolute_error_pct"]) for row in task_rows) * 1.04,
    )
    for axis, (result_scope, rows, label) in zip(
            axes,
            (
                ("real_kernel", kernel_rows, "Real kernels"),
                ("aggregate_task", task_rows, "Aggregate tasks"),
            ),
            strict=True):
        for hardware_id, title in HARDWARE:
            hardware_rows = [
                row for row in rows if row["hardware_id"] == hardware_id]
            values = np.sort(np.asarray([
                row["absolute_error_pct"] for row in hardware_rows]))
            if len(values) == 0:
                continue
            y = np.arange(1, len(values) + 1) / len(values) * 100.0
            axis.step(
                values, y, where="post",
                color=colors[hardware_id], label=title)
        axis.axvline(
            BUCKET_MAPE_LIMIT_PCT, color="#777777", linewidth=0.9,
            linestyle=":")
        axis.axvline(
            CASE_APE_LIMIT_PCT, color="#D55E00", linewidth=1.0,
            linestyle="--")
        worst = max(
            rows,
            key=lambda row: (
                float(row["absolute_error_pct"]),
                _observed_duration_ns(row, result_scope),
                row["case_id"],
            ),
        )
        ordinal = _kernel_ordinal_suffix(worst, result_scope)
        axis.scatter(
            [worst["absolute_error_pct"]], [100.0], s=24,
            color=colors[worst["hardware_id"]], marker="D", zorder=3)
        axis.annotate(
            f"max APE: {_plot_case_id(worst['case_id'])}{ordinal}\n"
            f"{worst['absolute_error_pct']:.1f}%",
            (worst["absolute_error_pct"], 100.0),
            xytext=(-6, -8), textcoords="offset points",
            horizontalalignment="right", verticalalignment="top",
            fontsize=6.5,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                        "color": "#666666"},
        )
        axis.set_xlim(0.0, error_upper)
        axis.set_ylim(0.0, 100.0)
        axis.set_title(label)
        axis.set_xlabel("Absolute percentage error (%)")
        axis.set_ylabel("Rows at or below error (%)")
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=3, frameon=False)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · complete error distribution on "
        f"a linear axis; guides at {BUCKET_MAPE_LIMIT_PCT:.0f}% and "
        f"{CASE_APE_LIMIT_PCT:.0f}%",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_residual_by_family(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharey=True)
    positions = np.arange(len(FAMILY_LABELS))
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        values = [
            [
                row["signed_error_pct"] for row in task_rows
                if row["hardware_id"] == hardware_id and row["family"] == family
            ]
            for family in FAMILY_LABELS
        ]
        axis.boxplot(values, positions=positions, widths=0.65, showfliers=True)
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.axhline(-CASE_APE_LIMIT_PCT, color="#D55E00",
                     linewidth=0.9, linestyle=":")
        axis.axhline(CASE_APE_LIMIT_PCT, color="#D55E00",
                     linewidth=0.9, linestyle=":")
        axis.set_xticks(positions, FAMILY_LABELS, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Signed duration error (%)")
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · task residuals by family")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_residual_vs_scale(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "nvidia_a800": "#0072B2",
        "metax_c500": "#D55E00",
        "ascend_910b": "#009E73",
    }
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.3), sharey=True)
    for axis, family in zip(axes, FAMILY_LABELS, strict=True):
        for hardware_id, title in HARDWARE:
            selected = [
                row for row in task_rows
                if row["hardware_id"] == hardware_id and row["family"] == family
            ]
            axis.scatter(
                [row["scale"] for row in selected],
                [row["signed_error_pct"] for row in selected],
                s=10, alpha=0.4, edgecolors="none", color=colors[hardware_id],
                label=title, rasterized=True,
            )
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_title(family)
        axis.set_xlabel("Canonical workload scale")
        axis.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("Signed duration error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · task residual versus workload "
        "scale",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_target_traffic_ratios(
        comparison_rows, path: Path, *, hardware_id: str, family: str,
        title: str, component_labels: Sequence[tuple[str, str]],
        category_order: Sequence[str], category_for_variant,
        logarithmic_ratio_axis: bool) -> None:
    """Plot traffic error itself so log-log coordinate distance cannot hide it."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    colors = {
        category: color for category, color in zip(
            category_order,
            ("#D55E00", "#0072B2", "#999999"),
            strict=True,
        )
    }
    markers = {
        category: marker for category, marker in zip(
            category_order, ("D", "s", "o"), strict=True)
    }
    figure, raw_axes = plt.subplots(
        1, len(component_labels),
        figsize=(7.1 * len(component_labels), 5.2), sharex=True)
    axes = np.atleast_1d(raw_axes)
    legend_handles = []
    legend_labels = []
    for axis, (component, component_label) in zip(
            axes, component_labels, strict=True):
        selected = [
            row for row in comparison_rows
            if row["hardware_id"] == hardware_id
            and row["family"] == family
            and row["component"] == component
            and row["predicted"] > 0 and row["observed"] > 0
        ]
        if not selected:
            raise RuntimeError(
                f"no traffic comparison rows for {hardware_id}/{family}/"
                f"{component}")
        severe = [
            row for row in selected
            if abs(row["predicted"] / row["observed"] - 1.0)
            > CASE_APE_LIMIT_PCT / 100.0
        ]
        for category in category_order:
            subset = [
                row for row in selected
                if category_for_variant(row["variant"]) == category
            ]
            if not subset:
                continue
            artist = axis.scatter(
                [row["observed"] for row in subset],
                [row["predicted"] / row["observed"] for row in subset],
                s=18 if category != category_order[-1] else 10,
                alpha=0.7 if category != category_order[-1] else 0.3,
                edgecolors="none", color=colors[category],
                marker=markers[category], label=category, rasterized=True,
            )
            if category not in legend_labels:
                legend_handles.append(artist)
                legend_labels.append(category)
        axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.08)
        axis.axhline(0.6, color="#666666", linewidth=0.8, linestyle=":")
        axis.axhline(1.0, color="#333333", linewidth=1.1, linestyle="--")
        axis.axhline(1.4, color="#666666", linewidth=0.8, linestyle=":")
        axis.set_xscale("log")
        if logarithmic_ratio_axis:
            axis.set_yscale("log")
        else:
            axis.set_ylim(0.0, 2.05)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}×"))
        axis.set_title(
            f"{component_label}\n{len(severe)}/{len(selected)} kernels "
            f">{CASE_APE_LIMIT_PCT:.0f}% off")
        axis.set_xlabel("Observed interface work")
        axis.set_ylabel("Predicted ÷ observed traffic")
        axis.grid(True, which="major", alpha=0.25)
        observed_values = [float(row["observed"]) for row in selected]
        observed_midpoint = math.sqrt(
            min(observed_values) * max(observed_values))
        labelled = []
        for category in category_order[:-1]:
            category_rows = [
                row for row in severe
                if category_for_variant(row["variant"]) == category
            ]
            if category_rows:
                labelled.append(max(
                    category_rows, key=lambda row: float(row["observed"])))
        for index, row in enumerate(labelled):
            ratio = row["predicted"] / row["observed"]
            observed_work = float(row["observed"])
            case_label = (
                row["case_id"]
                .removeprefix("FUSEDMOE_")
                .removeprefix("VECTOR_ROPE_")
                .removesuffix("_B1")
                .replace("_TOPK", "_K")
                .replace("_INT", "_I")
            )
            label = (
                f"{case_label}"
                f" / {row['variant']} / {ratio:.3g}×")
            place_left = observed_work > observed_midpoint
            axis.annotate(
                label,
                (observed_work, ratio),
                xytext=(-5 if place_left else 5, 6 + 13 * index),
                textcoords="offset points",
                horizontalalignment="right" if place_left else "left",
                fontsize=6.5,
                arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                            "color": "#666666"},
            )
    figure.suptitle(title)
    figure.legend(
        legend_handles, legend_labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.93), ncol=len(legend_labels), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_c500_vector_traffic_ratios(comparison_rows, path: Path) -> None:
    def category(variant: str) -> str:
        if variant in ("rope", "rotary"):
            return "RoPE"
        elif variant in ("rmsnorm", "add_rmsnorm"):
            return "RMSNorm"
        elif variant in (
                "activation", "silu", "kv_cache", "kvcache",
                "paged_kv_cache",
                "elementwise", "elementwise_pytorch_vec4"):
            return "Other Vector"
        else:
            raise ValueError(f"unknown C500 Vector variant: {variant!r}")

    _plot_target_traffic_ratios(
        comparison_rows,
        path,
        hardware_id="metax_c500",
        family="Vector",
        title=(
            "PROVENANCE-UNVERIFIED DIAGNOSTIC · MetaX C500 Vector "
            "directional traffic versus historical MACA 3.2.1 observation"),
        component_labels=(
            ("traffic:l2c_source_read", "L2C source-side read bytes"),
            ("traffic:downstream_global_read", "Downstream global read bytes"),
            ("traffic:downstream_global_write", "Downstream global write bytes"),
            ("demand:dram", "Downstream global read + write bytes"),
        ),
        category_order=("RoPE", "RMSNorm", "Other Vector"),
        category_for_variant=category,
        logarithmic_ratio_axis=False,
    )


def _plot_ascend_moe_traffic_ratios(replay_rows, path: Path) -> None:
    """Plot the two independently comparable GroupedMatmul interfaces."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if len(replay_rows) != 352:
        raise RuntimeError(
            "Ascend GroupedMatmul traffic plot requires 352 replay rows")

    colors = {"w1": "#D55E00", "w2": "#0072B2"}
    markers = {"w1": "D", "w2": "s"}
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=True)
    for variant in ("w1", "w2"):
        replay = [row for row in replay_rows if row["variant"] == variant]
        axes[0].scatter(
            [row["predicted_mte2_l2_read_bytes"] for row in replay],
            [row["read_predicted_over_observed"]
             for row in replay],
            s=18, alpha=0.7, edgecolors="none", color=colors[variant],
            marker=markers[variant], label=f"GroupedMatmul {variant}",
            rasterized=True)
        axes[1].scatter(
            [row["predicted_aic_output_l2_write_bytes"] for row in replay],
            [row["output_predicted_over_observed"] for row in replay],
            s=18, alpha=0.7, edgecolors="none", color=colors[variant],
            marker=markers[variant], label=f"GroupedMatmul {variant}",
            rasterized=True)

    read_within_five_pct = sum(
        row["read_absolute_error_pct"] <= 5.0 for row in replay_rows)
    output_within_twenty_pct = sum(
        row["output_absolute_error_pct"] <= 20.0 for row in replay_rows)
    mean_read_error = float(np.mean([
        row["read_absolute_error_pct"] for row in replay_rows]))
    mean_output_error = float(np.mean([
        row["output_absolute_error_pct"] for row in replay_rows]))
    axes[0].set_title(
        "Production MTE2 read → observed GM-to-L1\n"
        f"{read_within_five_pct}/352 within 5%; mean {mean_read_error:.2f}%")
    axes[1].set_title(
        "Production AIC output → observed write interface\n"
        f"{output_within_twenty_pct}/352 within 20%; mean {mean_output_error:.2f}%")
    for axis in axes:
        axis.set_xscale("log")
        axis.axhline(1.0, color="#333333", linewidth=1.1, linestyle="--")
        axis.set_xlabel("Modeled interface bytes")
        axis.set_ylabel("Predicted ÷ observed interface bytes")
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}×"))
        axis.grid(True, which="major", alpha=0.25)
    for axis in axes:
        axis.set_ylim(0.78, 1.22)
        axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.08)
        axis.axhline(0.8, color="#666666", linewidth=0.8, linestyle=":")
        axis.axhline(1.2, color="#666666", linewidth=0.8, linestyle=":")

    for index, variant in enumerate(("w1", "w2")):
        row = max(
            (item for item in replay_rows if item["variant"] == variant),
            key=lambda item: float(item["predicted_mte2_l2_read_bytes"]))
        ratio = row["read_predicted_over_observed"]
        modeled_bytes = row["predicted_mte2_l2_read_bytes"]
        axes[0].annotate(
            f"largest {variant}: {modeled_bytes / 2**20:.1f} MiB, "
            f"{ratio:.4f}×",
            (modeled_bytes, ratio),
            xytext=(-8, 12 + index * 14),
            textcoords="offset points", horizontalalignment="right",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                        "color": "#666666"},
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · Ascend 910B MoE production "
        "read/output demand versus counters")
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
        ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_c500_silu_mechanism(
        rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Plot demand closure and replay-only latency evidence without mixing clocks."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("C500 SiLU mechanism plot requires rows")
    duration_us = np.asarray([
        float(row["production_duration_ns"]) / 1000.0 for row in rows])
    predicted_ratio = np.asarray([
        float(row["predicted_over_production"]) for row in rows])
    waves = np.asarray([float(row["waves"]) for row in rows])
    summary = _c500_silu_mechanism_summary(rows)
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 10.2))

    duration_axis = axes[0][0]
    duration_axis.scatter(
        duration_us, predicted_ratio, s=22, alpha=0.72,
        color="#0072B2", edgecolors="none")
    duration_axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.10)
    duration_axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    duration_axis.axhline(0.6, color="#D55E00", linestyle=":", linewidth=1)
    duration_axis.axhline(1.4, color="#D55E00", linestyle=":", linewidth=1)
    duration_axis.set_xscale("log")
    duration_axis.set_ylim(0.0, 1.5)
    duration_axis.set_xlabel("Canonical task duration (µs)")
    duration_axis.set_ylabel("Forward ÷ production duration")
    duration_axis.set_title(
        "A · Final time: "
        f"{summary['underpredicted_count']}/{summary['case_count']} "
        f"underpredicted; MAPE {summary['duration_mape_pct']:.1f}%")
    duration_axis.grid(True, which="major", alpha=0.25)
    largest_delta_row = max(
        rows,
        key=lambda item: abs(
            float(item["predicted_duration_ns"])
            - float(item["production_duration_ns"])),
    )
    x_value = float(largest_delta_row["production_duration_ns"]) / 1000.0
    y_value = float(largest_delta_row["predicted_over_production"])
    duration_axis.annotate(
        f"largest absolute miss\n"
        f"{_plot_case_id(largest_delta_row['case_id'], 27)} · {y_value:.2f}×",
        (x_value, y_value), xytext=(-8, 14),
        textcoords="offset points", horizontalalignment="right",
        fontsize=7,
        arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                    "color": "#666666"})

    demand_axis = axes[0][1]
    instruction_rows = [
        row for row in rows
        if row["instruction_comparison_status"] == "comparable"
    ]
    if instruction_rows:
        demand_axis.scatter(
            [float(row["production_duration_ns"]) / 1000.0
             for row in instruction_rows],
            [float(row["instruction_predicted_over_observed"])
             for row in instruction_rows],
            marker="o", s=24, alpha=0.7, color="#0072B2",
            edgecolors="none", label="Instructions")
    demand_axis.scatter(
        duration_us,
        [float(row["traffic_predicted_over_observed"]) for row in rows],
        marker="^", s=27, alpha=0.7, color="#D55E00",
        edgecolors="none", label="Global bytes")
    demand_axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.10)
    demand_axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    demand_axis.axhline(0.6, color="#D55E00", linestyle=":", linewidth=1)
    demand_axis.axhline(1.4, color="#D55E00", linestyle=":", linewidth=1)
    demand_axis.set_xscale("log")
    demand_axis.set_ylim(0.55, 1.45)
    demand_axis.set_xlabel("Production mcTracer duration (µs)")
    demand_axis.set_ylabel("Predicted demand ÷ mcProfiler counter")
    if summary["instruction_comparison_status"] == "comparable":
        instruction_title = (
            f"instruction MAPE {summary['instruction_mape_pct']:.3f}%")
    elif summary["instruction_comparison_status"] == "unavailable":
        instruction_title = "instruction comparison unavailable"
    else:
        raise ValueError(
            "unsupported C500 SiLU instruction comparison status: "
            f"{summary['instruction_comparison_status']!r}")
    demand_axis.set_title(
        f"B · Work amount: {instruction_title}, bytes "
        f"MAPE {summary['traffic_mape_pct']:.2f}%")
    demand_axis.legend(frameon=False)
    demand_axis.grid(True, which="major", alpha=0.25)

    replay_axis = axes[1][0]
    replay_axis.scatter(
        waves,
        [float(row["instrumented_mean_active_aps"]) for row in rows],
        s=23, alpha=0.72, color="#009E73", edgecolors="none")
    replay_axis.axhline(
        104.0, color="#333333", linestyle="--", linewidth=1,
        label="104 physical APs")
    replay_axis.set_xscale("log")
    replay_axis.set_ylim(0.0, 110.0)
    replay_axis.set_xlabel("Dispatched waves in mcProfiler replay")
    replay_axis.set_ylabel("Time-averaged active AP count in that replay")
    replay_axis.set_title(
        "C · Same-replay activity only; not production utilization")
    replay_axis.legend(frameon=False)
    replay_axis.grid(True, which="major", alpha=0.25)

    wait_axis = axes[1][1]
    colors = np.asarray([
        float(row["instrumented_dnoc_read_latency_cycles"])
        for row in rows])
    points = wait_axis.scatter(
        waves,
        [float(row["instrumented_average_wave_life_cycles"])
         for row in rows],
        c=colors, cmap="viridis", s=25, alpha=0.78, edgecolors="none")
    wait_axis.set_xscale("log")
    wait_axis.set_xlabel("Dispatched waves in mcProfiler replay")
    wait_axis.set_ylabel("Average wave lifetime (XCORE cycles)")
    wait_axis.set_title(
        "D · Replay shows long-lived waves and read latency")
    wait_axis.grid(True, which="major", alpha=0.25)
    colorbar = figure.colorbar(points, ax=wait_axis)
    colorbar.set_label("DNoC read latency (cycles)")

    figure.suptitle(
        "MetaX C500 SiLU mechanism boundary · production time and "
        "instrumented replay are shown separately")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_vector_variant_residual(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canonical_variants = (
        "rmsnorm", "add_rmsnorm", "silu", "rope", "kv_cache",
        "elementwise",
    )
    aliases = {
        "rmsnorm": "rmsnorm",
        "add_rmsnorm": "add_rmsnorm",
        "silu": "silu",
        "rope": "rope",
        "rotary": "rope",
        "kv_cache": "kv_cache",
        "kvcache": "kv_cache",
        "paged_kv_cache": "kv_cache",
        "elementwise": "elementwise",
        "elementwise_pytorch_vec4": "elementwise",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    positions = np.arange(len(canonical_variants))
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        selected = [
            row for row in task_rows
            if row["hardware_id"] == hardware_id and row["family"] == "Vector"
        ]
        unknown = {
            row["variant_signature"] for row in selected
            if row["variant_signature"] not in aliases
        }
        if unknown:
            raise ValueError(
                f"unknown Vector variant signatures: {sorted(unknown)}")
        values = [
            [
                row["signed_error_pct"] for row in selected
                if aliases[row["variant_signature"]] == variant
            ]
            for variant in canonical_variants
        ]
        if any(not group for group in values):
            raise RuntimeError(
                f"{hardware_id} lacks a canonical Vector variant")
        axis.boxplot(
            values, positions=positions, widths=0.65, showfliers=True)
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.axhline(-CASE_APE_LIMIT_PCT, color="#D55E00",
                     linewidth=0.9, linestyle=":")
        axis.axhline(CASE_APE_LIMIT_PCT, color="#D55E00",
                     linewidth=0.9, linestyle=":")
        axis.set_xticks(
            positions, canonical_variants, rotation=30, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Signed task-duration error (%)")
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · Vector task residuals by "
        "variant")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_residual_vs_derived_launch_rounds(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "GEMM": "#0072B2",
        "FlashAttention": "#009E73",
        "FusedMoE": "#D55E00",
        "Vector": "#CC79A7",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharey=True)
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        selected_hardware = [
            row for row in task_rows if row["hardware_id"] == hardware_id]
        for family in FAMILY_LABELS:
            selected = [
                row for row in selected_hardware if row["family"] == family]
            axis.scatter(
                [row["max_derived_launch_rounds"] for row in selected],
                [row["signed_error_pct"] for row in selected],
                s=10, alpha=0.4, edgecolors="none", color=colors[family],
                label=family, rasterized=True,
            )
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("Maximum derived grid rounds")
        axis.set_ylabel("Signed task-duration error (%)")
        axis.grid(True, which="major", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · task residual versus derived "
        "launch rounds",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_attention_split_residual(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharey=True)
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        selected = [
            row for row in task_rows
            if row["hardware_id"] == hardware_id
            and row["family"] == "FlashAttention"
            and row["executed_attention_splits"] is not None
        ]
        if not selected:
            axis.text(
                0.5, 0.5, "No externally comparable split contract",
                ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        axis.scatter(
            [row["executed_attention_splits"] for row in selected],
            [row["signed_error_pct"] for row in selected],
            s=12, alpha=0.45, edgecolors="none", rasterized=True,
        )
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_title(title)
        axis.set_xlabel("Observed executed KV splits")
        axis.set_ylabel("Signed task-duration error (%)")
        axis.grid(True, which="major", alpha=0.3)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · FlashAttention task residual "
        "versus executed split count")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_bottlenecks(kernel_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ("launch", "pipeline", "compute", "memory")

    def category(name: str) -> str:
        if name == "launch":
            return "launch"
        elif name == "pipeline":
            return "pipeline"
        elif name.startswith("compute") or name in {
                "tensor_pipe", "vector_pipe", "fma_sfu_pipe", "alu_pipe",
                "warp_issue", "scalar_aiv"}:
            return "compute"
        return "memory"

    labels = [
        f"{hardware_id}\n{family}"
        for hardware_id, _ in HARDWARE for family in FAMILY_LABELS
    ]
    counts = {name: [] for name in categories}
    unavailable_indices = []
    for hardware_id, _ in HARDWARE:
        for family in FAMILY_LABELS:
            selected = [
                row for row in kernel_rows
                if row["hardware_id"] == hardware_id and row["family"] == family
            ]
            total = len(selected)
            if total == 0:
                unavailable_indices.append(len(counts[categories[0]]))
                for name in categories:
                    counts[name].append(0.0)
                continue
            grouped = {name: 0 for name in categories}
            for row in selected:
                name = category(row["bottleneck"])
                grouped[name] += 1
            for name in categories:
                counts[name].append(grouped[name] / total * 100.0)
    figure, axis = plt.subplots(figsize=(14.2, 5.1))
    bottom = np.zeros(len(labels))
    for name in categories:
        values = np.asarray(counts[name])
        axis.bar(labels, values, bottom=bottom, label=name)
        bottom += values
    axis.set_ylabel("Kernel share (%)")
    axis.set_ylim(0, 100)
    for index in unavailable_indices:
        axis.text(
            index, 3.0, "n/a\nOP timing", ha="center", va="bottom",
            fontsize=7, color="#666666")
    axis.legend(ncol=4, frameon=False, loc="upper center")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_support_coverage(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = ("complete", "partial", "aggregate_only", "compatibility")
    labels = [
        f"{title}\n{family}"
        for _, title in HARDWARE for family in FAMILY_LABELS
    ]
    counts = {level: [] for level in levels}
    for hardware_id, _ in HARDWARE:
        for family in FAMILY_LABELS:
            selected = [
                row for row in task_rows
                if row["hardware_id"] == hardware_id and row["family"] == family
            ]
            if not selected:
                raise RuntimeError(
                    f"no support rows for {hardware_id}/{family}")
            for level in levels:
                counts[level].append(
                    sum(row["support_level"] == level for row in selected)
                    / len(selected) * 100.0)
    figure, axis = plt.subplots(figsize=(14.2, 5.1))
    bottom = np.zeros(len(labels))
    for level in levels:
        values = np.asarray(counts[level])
        axis.bar(labels, values, bottom=bottom, label=level)
        bottom += values
    axis.set_ylabel("Task share (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Forward diagnostic support levels (not validation status)")
    axis.legend(ncol=4, frameon=False, loc="upper center")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _render_plot_call(call) -> str:
    """Render one independent report figure in a worker process."""

    plot, arguments, path = call
    plot(*arguments)
    return str(path)


def _write_markdown_summary(
        path: Path, baseline_rows, bucket_rows, accuracy_gate,
        comparison_summary, case_impact_rows, c500_silu_rows, artifacts,
        before_after_summary, before_after_bucket_rows,
        before_results_dir) -> None:
    comparison_reasons = sorted({
        row["comparison_reason"]
        for row in accuracy_gate["comparison_row_violations"]
    })
    if not comparison_reasons:
        comparison_reason_text = "none"
    else:
        comparison_reason_text = ", ".join(
            f"`{reason}`" for reason in comparison_reasons)
    lines = [
        "# Kernel 机制验证：自动生成的数据摘要",
        "",
        "本文件只陈述由当前 Forward 与 canonical observation 自动计算出的结果，不用最终时延反推或调节模型参数。机制因果结论需要结合审阅报告与独立证据判断。",
        "",
        f"**重要：当前 canonical duration 有数值，因此可以用于定位离群 case；但采集时的 runtime、source snapshot、runner/operator binary 身份未记录，所以这些数值都是 provenance-unverified diagnostic，不进入正式 {CASE_APE_LIMIT_PCT:.0f}%/{BUCKET_MAPE_LIMIT_PCT:.0f}% 验收分母。**",
        "",
        f"当前正式比较缺口：{comparison_reason_text}。",
        "",
        "## 先看口径",
        "",
        "| 层级 | 一行代表什么 | 实测字段 | 预测字段 | 单行门槛 | 物理桶门槛 |",
        "|---|---|---|---|---:|---:|",
        f"| `real_kernel` | profiler 中一次有独立时延的真实 kernel invocation | `actual_ns` | 单 kernel Forward | APE ≤ {CASE_APE_LIMIT_PCT:.0f}% | 桶内算术平均 MAPE ≤ {BUCKET_MAPE_LIMIT_PCT:.0f}% |",
        f"| `aggregate_task` | 一个 dataset task；真实值保留 canonical OP/kernel-sum 边界 | `observed_ns` | `{TASK_AGGREGATION_CONTRACT}` | APE ≤ {CASE_APE_LIMIT_PCT:.0f}% | 桶内算术平均 MAPE ≤ {BUCKET_MAPE_LIMIT_PCT:.0f}% |",
        "",
        "## 精度门禁",
        "",
        f"精度门禁：**{'通过' if accuracy_gate['accuracy_gate_passed'] else '未通过'}**。完整机制闭环门禁：**{'通过' if accuracy_gate['model_closure_gate_passed'] else '未通过'}**。每个物理桶必须同时满足 MAPE ≤ {BUCKET_MAPE_LIMIT_PCT:.0f}% 和桶内每一行 APE ≤ {CASE_APE_LIMIT_PCT:.0f}%；完整闭环还要求桶内 Forward support 为 100%。",
        "",
        f"| 层级 | 总行数 | 正式可比行 | Unobserved 行 | 桶数 | 正式口径覆盖通过桶 | 正式精度通过桶 | 诊断 APE > {CASE_APE_LIMIT_PCT:.0f}% 行 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("real_kernel", "aggregate_task"):
        selected = [
            row for row in bucket_rows if row["result_scope"] == scope]
        scope_gate = next(
            row for row in accuracy_gate["scopes"]
            if row["result_scope"] == scope)
        accuracy_passed = sum(
            row["accuracy_bucket_gate_passed"] for row in selected)
        comparison_passed = sum(
            row["comparison_coverage_gate_passed"] for row in selected)
        lines.append(
            f"| {scope} | {scope_gate['row_count']} | "
            f"{scope_gate['comparable_row_count']} | "
            f"{scope_gate['unobserved_row_count']} | {len(selected)} | "
            f"{comparison_passed} | {accuracy_passed} | "
            f"{scope_gate['diagnostic_case_violation_count']} |")
    if before_after_summary is not None:
        if before_after_bucket_rows is None or before_results_dir is None:
            raise ValueError(
                "before/after summary requires bucket rows and run identity")
        lines.extend([
            "",
            "## 同一批 case、同一套冻结物理桶的修前/修后诊断",
            "",
            f"修前预测读取自 `{before_results_dir}`。逐行先按 hardware、family、case 和 kernel ordinal（task 不含 ordinal）做一对一校验，再沿用本次 run 的冻结物理桶；实测时延和静态 launch 身份任何不一致都会直接报错。当前 capture 仍缺软件与 binary 身份，所以以下是诊断结果，不是正式验收结论。",
            "",
            f"| 层级 | Hardware | Family | Rows | 修前 MAPE | 修后 MAPE | 变化 | 修前最大 APE | 修后最大 APE | 修前 >{CASE_APE_LIMIT_PCT:.0f}% | 修后 >{CASE_APE_LIMIT_PCT:.0f}% |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in before_after_summary:
            if row["family"] == "ALL":
                continue
            lines.append(
                f"| {row['result_scope']} | {row['hardware_id']} | "
                f"{row['family']} | {row['row_count']} | "
                f"{row['before_mape_pct']:.2f}% | "
                f"{row['after_mape_pct']:.2f}% | "
                f"{row['mape_change_pct_points']:+.2f} pp | "
                f"{row['before_max_ape_pct']:.2f}% | "
                f"{row['after_max_ape_pct']:.2f}% | "
                f"{row['before_case_violation_count']} | "
                f"{row['after_case_violation_count']} |")
        lines.extend([
            "",
            f"| 层级 | 总桶数 | 诊断修前桶均值≤{BUCKET_MAPE_LIMIT_PCT:.0f}% | 诊断修后桶均值≤{BUCKET_MAPE_LIMIT_PCT:.0f}% | 诊断修前同时满足{BUCKET_MAPE_LIMIT_PCT:.0f}%/{CASE_APE_LIMIT_PCT:.0f}% | 诊断修后同时满足{BUCKET_MAPE_LIMIT_PCT:.0f}%/{CASE_APE_LIMIT_PCT:.0f}% |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for result_scope in ("real_kernel", "aggregate_task"):
            selected = [
                row for row in before_after_bucket_rows
                if row["result_scope"] == result_scope]
            lines.append(
                f"| {result_scope} | {len(selected)} | "
                f"{sum(row['before_diagnostic_mape_gate_passed'] for row in selected)} | "
                f"{sum(row['after_diagnostic_mape_gate_passed'] for row in selected)} | "
                f"{sum(row['before_diagnostic_accuracy_gate_passed'] for row in selected)} | "
                f"{sum(row['after_diagnostic_accuracy_gate_passed'] for row in selected)} |")
    lines.extend([
        "",
        "## 完整 Forward 基线",
    ])
    for scope, title in (
            ("real_kernel", "真实 kernel invocation 基线（`actual_ns`）"),
            (
                "aggregate_task",
                "聚合 task 基线（Forward kernel 预测串行求和；"
                "`observed_ns` 保留 canonical task timing）",
            )):
        lines.extend([
            "",
            f"### {title}",
            "",
            "| Hardware | Family | Rows | Comparable | Unobserved | Formal MAPE | Formal P90 | Formal Max | Diagnostic MAPE | Diagnostic P90 | Diagnostic Max | Diagnostic signed bias |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in baseline_rows:
            if row["result_scope"] != scope:
                continue
            lines.append(
                f"| {row['hardware_id']} | {row['family']} | "
                f"{row['row_count']} | {row['comparable_count']} | "
                f"{row['unobserved_count']} | "
                f"{_format_percentage(row['mape_pct'])} | "
                f"{_format_percentage(row['p90_ape_pct'])} | "
                f"{_format_percentage(row['max_ape_pct'])} | "
                f"{_format_percentage(row['diagnostic_mape_pct'])} | "
                f"{_format_percentage(row['diagnostic_p90_ape_pct'])} | "
                f"{_format_percentage(row['diagnostic_max_ape_pct'])} | "
                f"{_format_percentage(row['diagnostic_mean_signed_error_pct'], signed=True)} |")
    lines.extend([
        "",
        f"## 诊断 APE 超过 {CASE_APE_LIMIT_PCT:.0f}% 且绝对时间影响最大的行",
        "",
        "这里保留 provenance 未验证的 duration 用于查问题，按 `|预测时延−实测时延|` 排序，不按百分比误差或实测时延排序，也不代表正式验收。完整清单见 `case_impact_ranking_over_30pct.csv`。",
        "",
        "| Rank | Scope | Hardware | Family | Case | Kernel | Measured | Predicted | Abs. delta | APE | Bucket MAPE |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in case_impact_rows[:20]:
        ordinal = (
            "—" if row["kernel_ordinal"] is None
            else str(row["kernel_ordinal"]))
        lines.append(
            f"| {row['impact_rank']} | {row['result_scope']} | "
            f"{row['hardware_id']} | {row['family']} | "
            f"`{row['case_id']}` | {ordinal} | "
            f"{row['measured_ns'] / 1.0e6:.3f} ms | "
            f"{row['predicted_ns'] / 1.0e6:.3f} ms | "
            f"{row['absolute_duration_error_ns'] / 1.0e6:.3f} ms | "
            f"{row['absolute_error_pct']:.1f}% | "
            f"{row['bucket_mape_pct']:.1f}% |")
    c500_silu = _c500_silu_mechanism_summary(c500_silu_rows)
    if c500_silu["instruction_comparison_status"] == "comparable":
        instruction_summary = (
            f"指令数 MAPE 为 {c500_silu['instruction_mape_pct']:.6f}%"
            f"（{c500_silu['instruction_exact_count']}/"
            f"{c500_silu['instruction_comparable_count']} 完全一致，最大误差 "
            f"{c500_silu['instruction_max_ape_pct']:.6f}%）")
    elif c500_silu["instruction_comparison_status"] == "unavailable":
        instruction_summary = (
            "当前 Forward 未发布可与 mcProfiler 总指令数直接比较的静态"
            "指令下界，因此该比较明确记为 unavailable")
    else:
        raise ValueError(
            "unsupported C500 SiLU instruction comparison status: "
            f"{c500_silu['instruction_comparison_status']!r}")
    lines.extend([
        "",
        "## C500 SiLU：现在能确定什么",
        "",
        f"这 {c500_silu['case_count']} 个 case 中，Forward 有 {c500_silu['underpredicted_count']} 个预测偏小；MAPE 为 {c500_silu['duration_mape_pct']:.2f}%，最大 APE 为 {c500_silu['duration_max_ape_pct']:.2f}%，只有 {c500_silu['case_gate_passed_count']}/{c500_silu['case_count']} 满足单 case 不超过 {CASE_APE_LIMIT_PCT:.0f}% 的目标。绝对时间差最大的 case 是 `{c500_silu['largest_absolute_delta_case_id']}`：实测 {c500_silu['largest_absolute_delta_production_ns'] / 1000.0:.3f} µs，预测 {c500_silu['largest_absolute_delta_predicted_ns'] / 1000.0:.3f} µs，APE {c500_silu['largest_absolute_delta_ape_pct']:.2f}%。",
        "",
        f"工作量核验中，{instruction_summary}；全局内存流量 MAPE 为 {c500_silu['traffic_mape_pct']:.3f}%，最大误差 {c500_silu['traffic_max_ape_pct']:.3f}%。因此不能用不存在的指令比较替代物理证据，当前只对已发布的流量需求作直接核验。",
        "",
        f"mcProfiler 确实提供了这批重放的 XCORE 频率（{c500_silu['replay_core_clock_ghz_min']:.3f}–{c500_silu['replay_core_clock_ghz_max']:.3f} GHz）。只在同一次 mcProfiler 重放内部，用这个频率换算后，平均同时活跃 AP 数为 {c500_silu['replay_mean_active_aps_min']:.2f}–{c500_silu['replay_mean_active_aps_max']:.2f}/104，Average Wave Life 为 {c500_silu['replay_wave_life_cycles_min']:.0f}–{c500_silu['replay_wave_life_cycles_max']:.0f} cycles，DNoC read latency 为 {c500_silu['replay_dnoc_latency_cycles_min']:.1f}–{c500_silu['replay_dnoc_latency_cycles_max']:.1f} cycles，并存在 VLS pipeline/wdata stall。这证明 kernel 内有显著等待和依赖链，当前只按工作量除吞吐的模型漏掉了重要时间；但这些 counter 来自另一次插桩重放，不能除以 mcTracer 的 production duration，也不能直接充当 production 利用率或 production 服务时间。",
        "",
        f"读图方法：A、B 两图的纵轴都是倍率而不是 log；1.0 是完全一致，绿色区域是 ±{BUCKET_MAPE_LIMIT_PCT:.0f}%，橙色虚线是单 case ±{CASE_APE_LIMIT_PCT:.0f}% 边界。C、D 只描述同一次 mcProfiler 重放内部的并发和等待，不与 production 时间混算。完整逐 case 数值见 `c500_silu_mechanism.csv`。",
        "",
        "![C500 SiLU 机制边界](figures/c500-silu-mechanism-boundary.png)",
    ])
    lines.extend([
        "",
        "## Oracle 规范化后的 profiler 中间量诊断",
        "",
        "这里只列出字段语义已对齐且 provenance 种类为 `profiler` 或 `profiler_derived` 的量。但当前 counter capture 同样缺少 runtime/source/binary identity，所以正式指标为 `n/a`，数值只保留在 Diagnostic 列用于查问题。预测和观测同时为零的占位项、910B 未完成语义映射的原始 compute count 均不进入比率。",
        "",
        "| Hardware | Family | Component | Unit | Rows | Comparable | Formal MAPE | Diagnostic median pred/obs | Diagnostic P10 | Diagnostic P90 | Diagnostic MAPE | Diagnostic zero mismatch |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison_summary:
        if row["diagnostic_median_predicted_over_observed"] is None:
            diagnostic_ratio = diagnostic_p10 = diagnostic_p90 = diagnostic_mape = "n/a"
        else:
            diagnostic_ratio = (
                f"{row['diagnostic_median_predicted_over_observed']:.3f}")
            diagnostic_p10 = (
                f"{row['diagnostic_p10_predicted_over_observed']:.3f}")
            diagnostic_p90 = (
                f"{row['diagnostic_p90_predicted_over_observed']:.3f}")
            diagnostic_mape = _format_percentage(
                row["diagnostic_mape_pct"])
        lines.append(
            f"| {row['hardware_id']} | {row['family']} | "
            f"{row['component']} | {row['unit']} | "
            f"{row['row_count']} | {row['comparable_count']} | "
            f"{_format_percentage(row['mape_pct'])} | "
            f"{diagnostic_ratio} | {diagnostic_p10} | {diagnostic_p90} | "
            f"{diagnostic_mape} | "
            f"{row['diagnostic_zero_mismatch_count']} |")
    lines.extend(["", "## 生成产物", ""])
    for artifact in artifacts:
        lines.append(f"- `{artifact}`")
    path.write_text("\n".join(lines) + "\n")


def generate_report(
        output_dir: Path,
        jobs: int,
        before_results_dir: Path | None = None) -> Mapping[str, Any]:
    started_at = time.perf_counter()
    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise TypeError("jobs must be an integer")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    output_dir = _validate_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    figures_dir = output_dir / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_identities = {}
    lfs = _lfs_oids()
    for hardware_id, _ in HARDWARE:
        for family in DATASET_FAMILIES:
            dataset_path = Path(get_kernel_dataset_path(hardware_id, family))
            relative = str(dataset_path.relative_to(REPOSITORY_ROOT))
            if relative not in lfs:
                raise ValueError(
                    f"canonical dataset is not tracked by Git LFS: {relative}")
            dataset_identities[relative] = {
                "lfs_oid_sha256": lfs[relative],
                "worktree_sha256": _sha256(dataset_path),
            }
    launch_data_identities = {}
    for relative, oid in sorted(lfs.items()):
        if not relative.startswith("launch_data/"):
            continue
        launch_path = REPOSITORY_ROOT / relative
        launch_data_identities[relative] = {
            "lfs_oid_sha256": oid,
            "worktree_sha256": _sha256(launch_path),
        }
    config_identities = {
        str(Path(get_hardware_config_path(hardware_id)).relative_to(REPOSITORY_ROOT)): (
            _sha256(Path(get_hardware_config_path(hardware_id))))
        for hardware_id, _ in HARDWARE
    }
    before_manifest = None
    before_manifest_path = None
    before_manifest_sha256 = None
    if before_results_dir is not None:
        before_results_dir = _validate_output_directory(before_results_dir)
        if before_results_dir == output_dir:
            raise ValueError(
                "before-results directory must differ from output directory")
        before_manifest_path = before_results_dir / "manifest.json"
        if not before_manifest_path.is_file():
            raise FileNotFoundError(
                f"before-results manifest is missing: {before_manifest_path}")
        before_manifest = json.loads(before_manifest_path.read_text())
        for field, current_identity in (
                ("dataset_identities", dataset_identities),
                ("launch_data_identities", launch_data_identities),
                ("hardware_config_sha256", config_identities)):
            if before_manifest[field] != current_identity:
                raise ValueError(
                    f"before-results {field} differs from the current run")
        _verify_manifest_artifact(
            before_results_dir, before_manifest, "kernel_results.csv")
        _verify_manifest_artifact(
            before_results_dir, before_manifest, "task_results.csv")
        before_manifest_sha256 = _sha256(before_manifest_path)

    documents = _collect_hardware_matrix(jobs)
    raw_paths = []
    for document in documents:
        raw_path = raw_dir / f"{document['hardware_id']}.json"
        raw_path.write_text(
            json.dumps(_json_value(document), indent=2, sort_keys=True) + "\n")
        raw_paths.append(raw_path)

    task_rows = _task_rows(documents)
    kernel_rows = _kernel_rows(documents)
    counterfactual_rows = _counterfactual_rows(documents)
    counterfactual_summary = _counterfactual_summary_rows(
        counterfactual_rows, task_rows)
    bucket_rows = _bucket_rows(kernel_rows, task_rows)
    accuracy_gate = _accuracy_gate(bucket_rows, kernel_rows, task_rows)
    severe_duration = _severe_duration_rows(kernel_rows, task_rows)
    accuracy_violations = [
        row for row in _severe_duration_rows(
            kernel_rows, task_rows, threshold_pct=CASE_APE_LIMIT_PCT)
        if row["comparison_status"] == "comparable"
        and row["absolute_error_pct"] > CASE_APE_LIMIT_PCT
    ]
    case_impact = _case_impact_rows(
        bucket_rows, kernel_rows, task_rows)
    comparisons = _comparison_rows(documents)
    baseline = _baseline_rows(kernel_rows, task_rows)
    comparison_summary = _comparison_summary(comparisons)
    coverage = _coverage_rows(documents, comparisons)
    traffic_outliers = _traffic_outlier_rows(comparisons)
    ascend_moe_tile_replay = _ascend_moe_tile_replay_rows(documents)
    if len(ascend_moe_tile_replay) != 352:
        raise RuntimeError(
            "Ascend GroupedMatmul replay check expected 352 kernels, got "
            f"{len(ascend_moe_tile_replay)}")
    c500_silu_mechanism = _c500_silu_mechanism_rows(documents)
    if len(c500_silu_mechanism) != 56:
        raise RuntimeError(
            "C500 SiLU mechanism check expected 56 kernels, got "
            f"{len(c500_silu_mechanism)}")

    before_after_cases = None
    before_after_buckets = None
    before_after_summary = None
    before_after_join_audit = None

    _write_csv(output_dir / "task_results.csv", task_rows, tuple(task_rows[0]))
    _write_csv(output_dir / "kernel_results.csv", kernel_rows, tuple(kernel_rows[0]))
    if before_results_dir is not None:
        before_kernel_rows = _read_result_csv(
            before_results_dir / "kernel_results.csv",
            (
                "hardware_id", "family", "op_type", "case_id",
                "kernel_ordinal", "kernel_kind", "variant", "kernel_name",
                "actual_ns", "predicted_ns", "support_level",
                "launch_algorithm", "launch_grid_size", "launch_block_size",
                "launch_execution_variant", "launch_tile_shape",
                "launch_num_kv_splits",
            ),
        )
        before_task_rows = _read_result_csv(
            before_results_dir / "task_results.csv",
            (
                "hardware_id", "family", "case_id", "predicted_ns",
                "observed_ns", "support_level", "kernel_count",
                "task_execution_signature", "gemm_m", "gemm_n", "gemm_k",
                "query_length", "kv_length",
                "executed_attention_splits", "attention_layout",
                "moe_experts", "moe_topk", "moe_max_tokens_per_expert",
                "vector_tokens",
            ),
        )
        after_kernel_rows = _read_result_csv(
            output_dir / "kernel_results.csv",
            (
                "result_scope", "hardware_id", "family", "op_type",
                "case_id", "kernel_ordinal", "kernel_kind", "variant",
                "kernel_name", "actual_ns", "timing_source", "predicted_ns",
                "support_level", "launch_algorithm", "launch_grid_size",
                "launch_block_size", "launch_execution_variant",
                "launch_tile_shape", "launch_num_kv_splits", "bucket_schema",
                "physical_bucket_id", "physical_bucket_short_id",
                "bucket_dimensions", "bucket_basis", "comparison_status",
                "comparison_reason",
            ),
        )
        after_task_rows = _read_result_csv(
            output_dir / "task_results.csv",
            (
                "result_scope", "hardware_id", "family", "case_id",
                "predicted_ns", "observed_ns", "timing_source",
                "support_level", "kernel_count", "task_execution_signature",
                "gemm_m", "gemm_n", "gemm_k", "query_length", "kv_length",
                "executed_attention_splits", "attention_layout",
                "moe_experts", "moe_topk", "moe_max_tokens_per_expert",
                "vector_tokens", "bucket_schema", "physical_bucket_id",
                "physical_bucket_short_id", "bucket_dimensions",
                "bucket_basis", "comparison_status", "comparison_reason",
            ),
        )
        before_after_cases, before_after_join_audit = _before_after_case_rows(
            before_kernel_rows,
            before_task_rows,
            after_kernel_rows,
            after_task_rows,
        )
        before_after_buckets = _before_after_bucket_rows(before_after_cases)
        before_after_summary = _before_after_summary_rows(before_after_cases)
        if before_manifest is None or before_manifest_path is None:
            raise AssertionError("before-results manifest was not loaded")
        if "forward_source_worktree_diff" in before_manifest:
            before_forward_diff_field = "forward_source_worktree_diff"
        elif "forward_source_diff" in before_manifest:
            before_forward_diff_field = "forward_source_diff"
        else:
            raise ValueError(
                "before-results manifest has no Forward source diff identity")
        before_forward_diff = before_manifest[before_forward_diff_field]
        if not isinstance(before_forward_diff, str):
            raise TypeError(
                "before-results Forward source diff identity must be text")
        before_after_join_audit = {
            "before_results_directory": str(
                before_results_dir.relative_to(REPOSITORY_ROOT)),
            "before_manifest_sha256": before_manifest_sha256,
            "before_git_head": before_manifest["git_head"],
            "before_forward_diff_field": before_forward_diff_field,
            "before_forward_diff_sha256": hashlib.sha256(
                before_forward_diff.encode()).hexdigest(),
            "current_git_head": _git_output("rev-parse", "HEAD"),
            "current_forward_worktree_diff_sha256": hashlib.sha256(
                _git_output("diff", "--", "src").encode()).hexdigest(),
            "dataset_identities_equal": True,
            "launch_data_identities_equal": True,
            "hardware_config_identities_equal": True,
            **before_after_join_audit,
        }
        _write_csv(
            output_dir / "before_after_case_results.csv",
            before_after_cases,
            BEFORE_AFTER_CASE_FIELDS,
        )
        _write_csv(
            output_dir / "before_after_bucket_results.csv",
            before_after_buckets,
            BEFORE_AFTER_BUCKET_FIELDS,
        )
        _write_csv(
            output_dir / "before_after_summary.csv",
            before_after_summary,
            BEFORE_AFTER_SUMMARY_FIELDS,
        )
        (output_dir / "before_after_join_audit.json").write_text(
            json.dumps(
                before_after_join_audit, indent=2, sort_keys=True) + "\n")
    _write_csv(
        output_dir / "bucket_results.csv",
        bucket_rows,
        tuple(bucket_rows[0]),
    )
    _write_csv(
        output_dir / "severe_duration_cases_over_50pct.csv",
        severe_duration,
        SEVERE_DURATION_FIELDS,
    )
    _write_csv(
        output_dir / "case_accuracy_violations_over_30pct.csv",
        accuracy_violations,
        SEVERE_DURATION_FIELDS,
    )
    _write_csv(
        output_dir / "case_impact_ranking_over_30pct.csv",
        case_impact,
        CASE_IMPACT_FIELDS,
    )
    (output_dir / "physical_bucket_definitions.json").write_text(
        json.dumps(
            _physical_bucket_definitions(), indent=2, sort_keys=True)
        + "\n")
    (output_dir / "accuracy_gate.json").write_text(
        json.dumps(accuracy_gate, indent=2, sort_keys=True) + "\n")
    _write_csv(output_dir / "baseline.csv", baseline, tuple(baseline[0]))
    _write_csv(
        output_dir / "component_comparisons.csv",
        comparisons,
        tuple(comparisons[0]),
    )
    _write_csv(
        output_dir / "component_comparison_summary.csv",
        comparison_summary,
        tuple(comparison_summary[0]),
    )
    _write_csv(
        output_dir / "counterfactual_results.csv",
        counterfactual_rows,
        tuple(counterfactual_rows[0]),
    )
    _write_csv(
        output_dir / "counterfactual_summary.csv",
        counterfactual_summary,
        tuple(counterfactual_summary[0]),
    )
    _write_csv(output_dir / "evidence_coverage.csv", coverage, tuple(coverage[0]))
    _write_csv(
        output_dir / "traffic_outliers_over_50pct.csv",
        traffic_outliers,
        tuple(traffic_outliers[0]),
    )
    _write_csv(
        output_dir / "ascend_910b_moe_tile_replay_check.csv",
        ascend_moe_tile_replay,
        tuple(ascend_moe_tile_replay[0]),
    )
    _write_csv(
        output_dir / "c500_silu_mechanism.csv",
        c500_silu_mechanism,
        C500_SILU_MECHANISM_FIELDS,
    )

    figures = [
        figures_dir / "task-duration-overview-loglog.png",
        figures_dir / "final-residual-by-family.png",
        figures_dir / "residual-vs-workload-scale.png",
        figures_dir / "bottleneck-composition.png",
        figures_dir / "forward-support-coverage.png",
        figures_dir / "vector-variant-residual.png",
        figures_dir / "residual-vs-derived-launch-rounds.png",
        figures_dir / "attention-split-residual.png",
        figures_dir / "c500-vector-traffic-ratio-vs-observed-work.png",
        figures_dir / "ascend-910b-moe-l2-ratio-vs-interface-bytes.png",
        figures_dir / "kernel-duration-ratio-vs-observed.png",
        figures_dir / "task-duration-ratio-vs-observed.png",
        figures_dir / "physical-bucket-mape.png",
        figures_dir / "duration-ape-cdf.png",
        figures_dir / "case-impact-ranking.png",
        figures_dir / "c500-silu-mechanism-boundary.png",
    ]
    plot_calls = [
        (_plot_final_latency, (task_rows, figures[0]), figures[0]),
        (_plot_residual_by_family, (task_rows, figures[1]), figures[1]),
        (_plot_residual_vs_scale, (task_rows, figures[2]), figures[2]),
        (_plot_bottlenecks, (kernel_rows, figures[3]), figures[3]),
        (_plot_support_coverage, (task_rows, figures[4]), figures[4]),
        (_plot_vector_variant_residual, (task_rows, figures[5]), figures[5]),
        (_plot_residual_vs_derived_launch_rounds,
         (task_rows, figures[6]), figures[6]),
        (_plot_attention_split_residual, (task_rows, figures[7]), figures[7]),
        (_plot_c500_vector_traffic_ratios,
         (comparisons, figures[8]), figures[8]),
        (_plot_ascend_moe_traffic_ratios, (
            ascend_moe_tile_replay, figures[9]), figures[9]),
        (_plot_duration_ratio, (
            kernel_rows, figures[10], "real_kernel"), figures[10]),
        (_plot_duration_ratio, (
            task_rows, figures[11], "aggregate_task"), figures[11]),
        (_plot_bucket_accuracy, (bucket_rows, figures[12]), figures[12]),
        (_plot_ape_cdf, (kernel_rows, task_rows, figures[13]), figures[13]),
        (_plot_case_impact_ranking,
         (case_impact, figures[14]), figures[14]),
        (_plot_c500_silu_mechanism, (
            c500_silu_mechanism, figures[15]), figures[15]),
    ]
    if before_after_summary is not None:
        if before_after_buckets is None:
            raise AssertionError("before/after buckets were not generated")
        before_after_figures = (
            figures_dir / "before-after-family-mape.png",
            figures_dir / "before-after-bucket-mape.png",
        )
        plot_calls.extend((
            (_plot_before_after_family_accuracy, (
                before_after_summary, before_after_figures[0]),
             before_after_figures[0]),
            (_plot_before_after_bucket_accuracy, (
                before_after_buckets, before_after_figures[1]),
             before_after_figures[1]),
        ))
        figures.extend(before_after_figures)
    with ProcessPoolExecutor(max_workers=min(jobs, len(plot_calls))) as pool:
        rendered = tuple(pool.map(_render_plot_call, plot_calls))
    if rendered != tuple(str(path) for path in figures):
        raise RuntimeError("report figure workers returned out of order")
    import matplotlib

    generated_summary_path = output_dir / "generated_summary.md"
    manifest_path = output_dir / "manifest.json"
    artifact_paths = [
        *raw_paths,
        output_dir / "baseline.csv",
        output_dir / "task_results.csv",
        output_dir / "kernel_results.csv",
        output_dir / "bucket_results.csv",
        output_dir / "severe_duration_cases_over_50pct.csv",
        output_dir / "case_accuracy_violations_over_30pct.csv",
        output_dir / "case_impact_ranking_over_30pct.csv",
        output_dir / "physical_bucket_definitions.json",
        output_dir / "accuracy_gate.json",
        output_dir / "component_comparisons.csv",
        output_dir / "component_comparison_summary.csv",
        output_dir / "counterfactual_results.csv",
        output_dir / "counterfactual_summary.csv",
        output_dir / "evidence_coverage.csv",
        output_dir / "traffic_outliers_over_50pct.csv",
        output_dir / "ascend_910b_moe_tile_replay_check.csv",
        output_dir / "c500_silu_mechanism.csv",
        *figures,
        generated_summary_path,
        manifest_path,
    ]
    if before_after_cases is not None:
        artifact_paths.extend((
            output_dir / "before_after_case_results.csv",
            output_dir / "before_after_bucket_results.csv",
            output_dir / "before_after_summary.csv",
            output_dir / "before_after_join_audit.json",
        ))
    artifact_names = [
        str(path.relative_to(output_dir)) for path in artifact_paths]
    _write_markdown_summary(
        generated_summary_path,
        baseline,
        bucket_rows,
        accuracy_gate,
        comparison_summary,
        case_impact,
        c500_silu_mechanism,
        artifact_names,
        before_after_summary,
        before_after_buckets,
        (
            None if before_results_dir is None
            else str(before_results_dir.relative_to(REPOSITORY_ROOT))
        ),
    )
    artifact_sha256 = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in artifact_paths
        if path != manifest_path
    }
    command = (
        "python scripts/evaluation/kernel/kernel_mechanism_validation_report.py "
        f"--output-dir {output_dir.relative_to(REPOSITORY_ROOT)} --jobs {jobs}"
    )
    if before_results_dir is not None:
        command += (
            " --before-results-dir "
            f"{before_results_dir.relative_to(REPOSITORY_ROOT)}")
    manifest = {
        "schema_version": 8,
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_status_porcelain": _git_output("status", "--short"),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "command": command,
        "hardware_ids": [hardware_id for hardware_id, _ in HARDWARE],
        "dataset_families": list(DATASET_FAMILIES),
        "dataset_identities": dataset_identities,
        "launch_data_identities": launch_data_identities,
        "hardware_config_sha256": config_identities,
        "physical_bucket_schema": {
            "version": BUCKET_SCHEMA_VERSION,
            "definition_sha256": artifact_sha256[
                "physical_bucket_definitions.json"],
        },
        "comparison_baseline": (
            None if before_after_join_audit is None else {
                "results_directory": before_after_join_audit[
                    "before_results_directory"],
                "manifest_sha256": before_after_join_audit[
                    "before_manifest_sha256"],
                "git_head": before_after_join_audit["before_git_head"],
                "forward_diff_field": before_after_join_audit[
                    "before_forward_diff_field"],
                "forward_diff_sha256": before_after_join_audit[
                    "before_forward_diff_sha256"],
            }
        ),
        "report_source_sha256": {
            "dataset_pipeline/kernel_dataset/ascend_910b.py": _sha256(
                REPOSITORY_ROOT
                / "dataset_pipeline/kernel_dataset/ascend_910b.py"),
            "evaluation/kernel/evaluation_runner.py": _sha256(
                REPOSITORY_ROOT / "evaluation/kernel/evaluation_runner.py"),
            "evaluation/kernel/mechanism_validation_report.py": _sha256(
                REPOSITORY_ROOT
                / "evaluation/kernel/mechanism_validation_report.py"),
            "evaluation/kernel/observation_normalizer.py": _sha256(
                REPOSITORY_ROOT
                / "evaluation/kernel/observation_normalizer.py"),
            "evaluation/kernel/oracle_observation.py": _sha256(
                REPOSITORY_ROOT
                / "evaluation/kernel/oracle_observation.py"),
            "evaluation/kernel/profiler_observation.py": _sha256(
                REPOSITORY_ROOT
                / "evaluation/kernel/profiler_observation.py"),
            "scripts/evaluation/kernel/kernel_mechanism_validation_report.py": (
                _sha256(
                    REPOSITORY_ROOT
                    / "scripts/evaluation/kernel/"
                    "kernel_mechanism_validation_report.py")),
            "src/kernel/predictors/fused_moe.py": _sha256(
                REPOSITORY_ROOT / "src/kernel/predictors/fused_moe.py"),
            "src/kernel/predictors/vector.py": _sha256(
                REPOSITORY_ROOT / "src/kernel/predictors/vector.py"),
            "src/kernel/formula.py": _sha256(
                REPOSITORY_ROOT / "src/kernel/formula.py"),
            "src/kernel/launch_config/metax_c500/flash_attn_launch.py": (
                _sha256(
                    REPOSITORY_ROOT
                    / "src/kernel/launch_config/metax_c500/"
                    "flash_attn_launch.py")),
        },
        "forward_source_worktree_diff": _git_output(
            "diff", "--", "src"),
        "forward_source_index_diff": _git_output(
            "diff", "--cached", "--", "src"),
        "forward_source_untracked": _git_output(
            "ls-files", "--others", "--exclude-standard", "--", "src"),
        "hardware_execution_performed": False,
        "profiler_capture_performed": False,
        "artifact_sha256": artifact_sha256,
        "manifest_self_hash": (
            "omitted because a file cannot contain its own stable SHA256"),
        "elapsed_seconds_before_manifest_write": (
            time.perf_counter() - started_at),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "output_dir": str(output_dir),
        "task_count": len(task_rows),
        "kernel_count": len(kernel_rows),
        "bucket_count": len(bucket_rows),
        "accuracy_gate_passed": accuracy_gate["accuracy_gate_passed"],
        "model_closure_gate_passed": accuracy_gate[
            "model_closure_gate_passed"],
        "comparison_count": len(comparisons),
        "counterfactual_count": len(counterfactual_rows),
        "artifacts": artifact_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture and render cross-hardware kernel mechanism evidence "
            "without changing the production Forward model."))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--before-results-dir", type=Path)
    arguments = parser.parse_args()
    result = generate_report(
        arguments.output_dir,
        arguments.jobs,
        arguments.before_results_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
