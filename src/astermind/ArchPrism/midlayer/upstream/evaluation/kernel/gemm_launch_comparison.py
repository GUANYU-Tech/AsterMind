"""Compare observed and injected GEMM launch facts without conflating roles."""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable, Mapping

from evaluation.kernel.dataset import (
    get_hardware_config_path,
    load_kernel_dataset,
    parse_kernel_evaluation_timing,
)
from evaluation.kernel.observation_normalizer import (
    normalize_kernel_spec,
    observed_gemm_split_k,
    observed_gemm_split_k_partial_bytes,
    observed_gemm_split_k_partial_dram_bytes,
    validate_kernel_observation_sequence,
)
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config.select_launch_config import select_launch_config


_SUPPORTED_HARDWARE_IDS = (
    "nvidia_a800",
    "metax_c500",
    "ascend_910b",
)

_PHYSICAL_LAUNCH_FIELDS = (
    "grid",
    "block",
)


def _kernel_records(entry: Mapping, case_id: str) -> list[Mapping]:
    raw = entry["kernels"]
    if isinstance(raw, dict):
        records = list(raw.values())
    elif isinstance(raw, list):
        records = list(raw)
    else:
        raise TypeError(
            f"GEMM dataset entry {case_id!r} kernels must be a mapping or list")
    if not records:
        raise ValueError(f"GEMM dataset entry {case_id!r} has no kernels")
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError(
            f"GEMM dataset entry {case_id!r} contains a non-mapping kernel")
    return records


# The static-workload inputs each hardware's GEMM selector requires beyond the
# observed shape.  A hardware outside this table has no comparison contract.
_GEMM_INJECTED_ATTRIBUTES = {
    "nvidia_a800": {
        "has_bias": False,
        "launch_policy": "injected_launch",
    },
    "metax_c500": {
        "position": "generic",
        "launch_policy": "injected_launch",
    },
    "ascend_910b": {
        "launch_policy": "injected_launch",
    },
}


def _injected_attributes(hardware_id: str) -> Mapping[str, object]:
    """Return explicit static-workload inputs required by each GEMM selector."""

    if hardware_id not in _GEMM_INJECTED_ATTRIBUTES:
        raise ValueError(
            f"unsupported GEMM launch-comparison hardware: {hardware_id!r}")
    return dict(_GEMM_INJECTED_ATTRIBUTES[hardware_id])


def _physical_fields(launch) -> Mapping[str, object]:
    return {
        "grid": launch.grid,
        "block": launch.block,
    }


def _execution_plan(launch) -> Mapping[str, object]:
    tiling = launch.tiling
    if tiling is None:
        tiling_value = None
    else:
        tiling_value = {
            "tile_shape": dict(tiling.tile_shape),
            "pipeline_depth": tiling.pipeline_depth,
            "split_k": tiling.split_k,
        }
    return {
        "tiling": tiling_value,
        "parameters": dict(launch.parameters),
        "execution_variant": launch.execution_variant,
    }


def _with_injected_physical_fields(observed_spec, injected_launch):
    """Replace only physical launch metadata and retain the observed plan."""

    launch = dataclasses.replace(
        observed_spec.launch,
        grid=injected_launch.grid,
        block=injected_launch.block,
    )
    return dataclasses.replace(observed_spec, launch=launch)


def _with_probe_physical_fields(observed_spec):
    """Change every physical field without changing the execution plan."""

    observed = observed_spec.launch
    grid = (observed.grid[0] + 1, observed.grid[1] + 1, observed.grid[2] + 1)
    block = (observed.block[0] + 1, observed.block[1], observed.block[2])
    launch = dataclasses.replace(
        observed,
        grid=grid,
        block=block,
    )
    return dataclasses.replace(observed_spec, launch=launch)


def _with_injected_execution_plan(observed_spec, injected_launch):
    """Replace only tile/split/plan facts and retain physical metadata."""

    launch = dataclasses.replace(
        observed_spec.launch,
        tiling=injected_launch.tiling,
        parameters=injected_launch.parameters,
        execution_variant=injected_launch.execution_variant,
    )
    return dataclasses.replace(observed_spec, launch=launch)


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("mean requires at least one value")
    return math.fsum(materialized) / len(materialized)


def _mape(predictions: Iterable[float], actuals: Iterable[float]) -> float:
    pairs = tuple(zip(predictions, actuals, strict=True))
    if not pairs:
        raise ValueError("MAPE requires at least one prediction")
    return 100.0 * _mean(abs(predicted - actual) / actual
                         for predicted, actual in pairs)


def compare_gemm_launch_sources(hardware_id: str) -> Mapping[str, object]:
    """Compare physical launch metadata separately from execution-plan facts."""

    if hardware_id not in _SUPPORTED_HARDWARE_IDS:
        raise ValueError(
            f"hardware_id must be one of {_SUPPORTED_HARDWARE_IDS}, "
            f"got {hardware_id!r}")
    hardware = HWSpec(get_hardware_config_path(hardware_id))
    dataset = load_kernel_dataset(hardware_id, kernels=("gemm",))
    rows = []

    for case_id, entry in sorted(dataset.items()):
        op_info = entry["op_info"]
        if op_info["Type"] != "GEMM":
            raise ValueError(
                f"gemm.json entry {case_id!r} has type {op_info['Type']!r}")
        records = _kernel_records(entry, case_id)
        validate_kernel_observation_sequence(records, hardware_id)
        timing = parse_kernel_evaluation_timing(entry, records, hardware_id)
        split_k = observed_gemm_split_k(records)
        partial_bytes = observed_gemm_split_k_partial_bytes(records)
        partial_dram_bytes = observed_gemm_split_k_partial_dram_bytes(records)

        for kernel_index, record in enumerate(records):
            observed_spec = normalize_kernel_spec(
                op_info,
                record,
                hardware_id,
                hardware,
                kernel_ordinal=kernel_index,
                gemm_split_k=split_k,
                gemm_split_k_partial_bytes=partial_bytes,
                gemm_split_k_partial_dram_bytes=partial_dram_bytes,
            )
            if observed_spec.kind is not KernelKind.GEMM:
                continue
            observed_prediction = predict(
                observed_spec, hardware, method="midlayer").t_total_ns
            probe_spec = _with_probe_physical_fields(observed_spec)
            probe_prediction = predict(
                probe_spec, hardware, method="midlayer").t_total_ns
            actual_ns = timing.durations_ns[kernel_index]
            common_row = {
                "case_id": case_id,
                "kernel_index": kernel_index,
                "actual_ns": actual_ns,
                "observed_prediction_ns": observed_prediction,
                "physical_contract_probe_prediction_ns": probe_prediction,
                "physical_contract_probe_delta_percent": (
                    100.0 * (probe_prediction - observed_prediction)
                    / observed_prediction),
            }
            injected_launch = select_launch_config(
                hardware_id,
                KernelKind.GEMM,
                observed_spec.shape,
                observed_spec.dtype,
                observed_spec.variant,
                _injected_attributes(hardware_id),
                hardware=hardware,
            )

            physical_spec = _with_injected_physical_fields(
                observed_spec, injected_launch)
            physical_prediction = predict(
                physical_spec, hardware, method="midlayer").t_total_ns
            execution_plan_spec = _with_injected_execution_plan(
                observed_spec, injected_launch)
            execution_plan_prediction = predict(
                execution_plan_spec, hardware, method="midlayer").t_total_ns
            observed_physical = _physical_fields(observed_spec.launch)
            injected_physical = _physical_fields(injected_launch)
            observed_plan = _execution_plan(observed_spec.launch)
            injected_plan = _execution_plan(injected_launch)
            rows.append(common_row | {
                "injected_execution_plan_status": "available",
                "observed_physical_launch": observed_physical,
                "injected_physical_launch": injected_physical,
                "physical_fields_match": {
                    name: observed_physical[name] == injected_physical[name]
                    for name in observed_physical
                },
                "physical_only_prediction_ns": physical_prediction,
                "physical_only_prediction_delta_percent": (
                    100.0 * (physical_prediction - observed_prediction)
                    / observed_prediction),
                "observed_execution_plan": observed_plan,
                "injected_execution_plan": injected_plan,
                "execution_plan_matches": observed_plan == injected_plan,
                "injected_plan_prediction_ns": execution_plan_prediction,
                "injected_plan_prediction_delta_percent": (
                    100.0
                    * (execution_plan_prediction - observed_prediction)
                    / observed_prediction),
            })

    covered = [
        row for row in rows
        if row["injected_execution_plan_status"] == "available"]
    if not covered:
        raise ValueError(
            f"{hardware_id} has no GEMM cases covered by injected launch data")
    field_match_percent = {
        name: 100.0 * sum(
            row["physical_fields_match"][name] for row in covered)
        / len(covered)
        for name in _PHYSICAL_LAUNCH_FIELDS
    }
    contract_probe_deltas = [
        abs(row["physical_contract_probe_delta_percent"])
        for row in rows
    ]
    physical_deltas = [
        abs(row["physical_only_prediction_delta_percent"])
        for row in covered
    ]
    plan_deltas = [
        abs(row["injected_plan_prediction_delta_percent"])
        for row in covered
    ]
    observed_predictions = [
        row["observed_prediction_ns"] for row in covered]
    plan_predictions = [
        row["injected_plan_prediction_ns"] for row in covered]
    actuals = [row["actual_ns"] for row in covered]
    summary = {
        "hardware_id": hardware_id,
        "gemm_cases": len(dataset),
        "gemm_main_kernels": len(rows),
        "injected_execution_plan_covered": len(covered),
        "injected_execution_plan_unavailable": 0,
        "comparison_contract": {
            "physical_launch_fields": [
                "grid", "block",
            ],
            "physical_launch_role": (
                "diagnostic only; held separate from the GEMM performance "
                "model"),
            "execution_plan_fields": [
                "tiling", "parameters", "execution_variant"],
            "execution_plan_role": (
                "retained algorithmic inputs including tile and split-K"),
        },
        "physical_launch": {
            "all_kernel_contract_probe_count": len(rows),
            "all_kernel_contract_probe_max_abs_delta_percent": max(
                contract_probe_deltas),
            "all_kernel_contract_probe_predictions_identical": all(
                delta == 0.0 for delta in contract_probe_deltas),
            "observed_vs_injected_covered_count": len(covered),
            "field_match_percent": field_match_percent,
            "prediction_mean_abs_delta_percent": _mean(physical_deltas),
            "prediction_max_abs_delta_percent": max(physical_deltas),
            "all_covered_predictions_identical": all(
                delta == 0.0 for delta in physical_deltas),
        },
        "execution_plan": {
            "plan_match_percent": 100.0 * sum(
                row["execution_plan_matches"] for row in covered)
            / len(covered),
            "observed_plan_mape_percent_on_covered": _mape(
                observed_predictions, actuals),
            "injected_plan_mape_percent_on_covered": _mape(
                plan_predictions, actuals),
            "prediction_mean_abs_delta_percent": _mean(plan_deltas),
            "prediction_max_abs_delta_percent": max(plan_deltas),
        },
        "missing_injected_key_counts": {},
    }
    return {"summary": summary, "cases": rows}


def compare_all_gemm_launch_sources() -> Mapping[str, object]:
    return {
        hardware_id: compare_gemm_launch_sources(hardware_id)
        for hardware_id in _SUPPORTED_HARDWARE_IDS
    }
