"""Exact TTFT/TPOT attribution across fixed and dynamic Scheduler trajectories."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean, median
import tempfile
import time
from typing import Callable, Mapping, Sequence

import zstandard

from evaluation.serving.metrics import sharegpt_compact_evaluation as compact
from evaluation.serving.metrics import sharegpt_evaluation_report as sharegpt_report
from evaluation.serving.metrics.diagnostics.scheduling_trajectory_replay_report import (
    HARDWARE_ORDER,
    write_scheduler_depth_diagnostic_plots,
    write_scheduling_trajectory_replay_quantile_plots,
)
from evaluation.serving.simulation import simulation_runner
from evaluation.serving.traces import scheduling_trajectory_replay as replay
from src.kernel.launch_config.gemm_launch_selection import (
    gemm_launch_software_revision,
    load_gemm_launch_selection,
)
from src.serving.host_timing import load_host_launch_profile

from evaluation.json_shape import require_object as _mapping

from evaluation.json_shape import file_sha256 as _sha256

from evaluation.json_shape import require_finite_number as _finite

from evaluation.json_shape import write_json_atomic as _write_json_atomic


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_METRICS = ("ttft_ms", "tpot_ms")
_POINTS = ("O", "T", "D", "L", "S", "F_D", "F_L", "F_S")
_COMPONENTS = (
    "reconstruction",
    "device_direct",
    "device_feedback",
    "launch_direct",
    "launch_feedback",
    "scheduler_direct",
    "scheduler_feedback",
)
# Canonical GEMM launch artifact each hardware's launch selection reads.  H200
# projects the A800 launch package (src/kernel/launch_config/
# select_launch_config.py), so its preflight validates the A800 selector.
_GEMM_LAUNCH_SELECTION_SOURCE = {
    "nvidia_a800": "nvidia_a800",
    "metax_c500": "metax_c500",
    "ascend_910b": "ascend_910b",
    "nvidia_h200": "nvidia_a800",
}
def decompose_metric(points: Mapping[str, object]) -> dict[str, object]:
    """Return a signed additive decomposition that exactly closes at F_S - O."""

    if set(points) != set(_POINTS):
        raise ValueError("metric points must contain exactly O/T/D/L/S/F_D/F_L/F_S")
    values = {name: _finite(points[name], name) for name in _POINTS}
    observed = values["O"]
    if observed <= 0.0:
        raise ValueError("observed metric O must be positive")
    components = {
        "reconstruction": values["T"] - observed,
        "device_direct": values["D"] - values["T"],
        "device_feedback": values["F_D"] - values["D"],
        "launch_direct": values["L"] - values["D"],
        "launch_feedback": (
            values["F_L"] - values["F_D"]
            - (values["L"] - values["D"])
        ),
        "scheduler_direct": values["S"] - values["L"],
        "scheduler_feedback": (
            values["F_S"] - values["F_L"]
            - (values["S"] - values["L"])
        ),
    }
    final_error = values["F_S"] - observed
    closure_residual = final_error - sum(components.values())
    tolerance = max(1e-9, 1e-6 * max(abs(final_error), observed))
    if abs(closure_residual) > tolerance:
        raise ValueError(
            f"metric attribution does not close: residual={closure_residual}, "
            f"tolerance={tolerance}"
        )
    relative = {name: value / observed for name, value in components.items()}
    stable_share = abs(final_error) > max(1e-9, 1e-12 * observed)
    shares = {
        name: (value / final_error if stable_share else None)
        for name, value in components.items()
    }
    return {
        "points": values,
        "components": components,
        "component_over_observed": relative,
        "component_share_of_final_signed_error": shares,
        "final_signed_error": final_error,
        "final_signed_relative_error": final_error / observed,
        "final_absolute_percentage_error": 100.0 * abs(final_error) / observed,
        "final_prediction_over_observed": values["F_S"] / observed,
        "closure_residual": closure_residual,
        "closure_tolerance": tolerance,
    }


def _run_one_case(
    repository_root: str,
    config_path: str,
    candidate_case_path: str,
    cell_id: str,
    host_profile_path: str,
    source_commit: str,
    full_cell: Mapping[str, object],
    replay_output_path: str,
    dynamic_cache_dir: str,
    load_validated_case: Callable[[Path], object],
    case_cache_path: str,
) -> dict[str, object]:
    root = Path(repository_root)
    config = compact.load_phase1_config(config_path)
    cell = compact.select_phase1_cells(config, (cell_id,), all_cases=False)[0]
    case_path = Path(candidate_case_path)
    case = load_validated_case(case_path)
    compact._validate_case_binding(config, cell, case)
    if case.manifest["schema_version"] != 5:
        raise ValueError(f"{cell_id} candidate must use compact schema v5")
    configured_depth = compact.scheduler_pipeline_depth_from_validated_case(case)
    if configured_depth != 2:
        raise ValueError(f"{cell_id} requires dataset-native depth=2")

    observed = compact._observed_metrics(case)
    full_metrics = _validated_full_cell(full_cell, cell_id, observed)
    profile = load_host_launch_profile(Path(host_profile_path))
    if profile.hardware_id != cell.hardware.hardware_id:
        raise ValueError("Host profile hardware differs from cell hardware")

    original_trace = case_path / "scheduler_steps.jsonl.zst"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding="utf-8", delete=True
    ) as projected_trace:
        request_contract = replay.request_contract_from_compact_case(case)
        observed_structure = _project_v5_scheduler_trace(
            original_trace, projected_trace, request_contract
        )
        projected_trace.flush()
        decode_work_offset = observed_structure["decode_work_token_offset"]
        replay_contract = [dict(request) for request in request_contract]
        if decode_work_offset:
            for request in replay_contract:
                request["output_tokens"] += decode_work_offset
        replay_path = Path(replay_output_path)
        replay_result = None
        if replay_path.is_file():
            replay_result = _mapping(
                json.loads(replay_path.read_text()), "cached replay result"
            )
            if replay_result["cell_id"] != cell.cell_id:
                raise ValueError("cached replay cell identity differs")
            if replay_result["source_sha256"]["candidate_scheduler_trace"] != (
                    _sha256(original_trace)):
                raise ValueError("cached replay source hash differs")
            if (
                replay_result["schema_version"] < 5
                or "graph_execution" not in replay_result["predictor_source"]
                or any(
                    "step_fingerprint" not in prediction
                    for prediction in replay_result["step_comparison"]
                )
            ):
                replay_result = None
        if replay_result is None:
            replay_result = replay.analyze_scheduling_trajectory_replay(
                root,
                config.source_path,
                cell.cell_id,
                Path(projected_trace.name),
                replay_contract,
                cell.model.model_config_sha256,
                scheduler_schema_version=observed_structure[
                    "scheduler_schema_version"
                ],
                host_launch_profile=profile,
                execution_mode="accelerated",
                pipeline_depth=configured_depth,
                pipeline_depth_identity={
                    "configured_depth": configured_depth,
                    "resolved_depth": configured_depth,
                    "mode": "dataset_native",
                    "evidence": case.manifest["runtime"][
                        "scheduler_pipeline_depth_evidence"
                    ],
                },
            )
            if decode_work_offset:
                _restore_client_visible_tpot(replay_result, request_contract)
            replay_result["source_sha256"]["candidate_scheduler_trace"] = (
                _sha256(original_trace)
            )
            replay_result["source_sha256"]["projected_scheduler_trace"] = (
                replay_result["source_sha256"]["scheduler_trace"]
            )
            replay_result["contract"]["scheduler_trace_projection"] = (
                "canonical_schema_preserving_graph_identity_without_duration_consumption"
            )
            _write_json_atomic(replay_path, replay_result)

    dynamic_modes = {}
    for mode in ("device_only", "device_plus_launch"):
        cache_path = Path(dynamic_cache_dir) / (
            f"{cell.cell_id.replace('/', '__')}--{mode}.json"
        )
        if cache_path.is_file():
            cached = _mapping(
                json.loads(cache_path.read_text()), "cached dynamic mode"
            )
            if (cached["schema_version"] != 1
                    or cached["kind"] != "scheduler_depth_dynamic_mode"
                    or cached["source_commit"] != source_commit
                    or cached["cell_id"] != cell.cell_id
                    or cached["mode"] != mode):
                raise ValueError(f"cached dynamic mode identity differs: {cache_path}")
            dynamic_modes[mode] = _mapping(
                cached["result"], "cached dynamic result"
            )
            continue
        if mode == "device_only":
            profile_path = None
        elif mode == "device_plus_launch":
            profile_path = host_profile_path
        else:
            raise ValueError(f"unsupported dynamic timing mode: {mode!r}")
        argv = compact.build_simulation_argv(
            root,
            config,
            cell,
            case,
            execution_mode="accelerated",
            serving_timing_mode=mode,
            host_launch_profile_path=profile_path,
        )
        argv.extend(("--step-retention", "full"))
        payload = simulation_runner.run_cli_args(argv)
        compact._validate_completed_payload(
            root,
            config,
            cell,
            case,
            payload,
            source_commit,
            "accelerated",
            expected_serving_timing_mode=mode,
        )
        comparison = compact._compare_prediction(
            observed, payload, len(case.requests)
        )
        host_timing = compact._validated_host_timing_payload(
            payload, mode, cell.hardware.hardware_id
        )
        dynamic_modes[mode] = {
            "metrics": {
                metric: {
                    "observed": comparison["latency"][metric]["mean"]["observed"],
                    "predicted": comparison["latency"][metric]["mean"]["predicted"],
                }
                for metric in _METRICS
            },
            "structure": _dynamic_structure(host_timing, payload["metrics"]),
            "forward_argv_sha256": _canonical_sha256(argv),
        }
        _write_json_atomic(cache_path, {
            "schema_version": 1,
            "kind": "scheduler_depth_dynamic_mode",
            "source_commit": source_commit,
            "cell_id": cell.cell_id,
            "mode": mode,
            "result": dynamic_modes[mode],
        })
        del payload
        gc.collect()

    replay_modes = replay_result["scheduling_trajectory_replay_modes"]
    points_by_metric = {}
    for metric in _METRICS:
        points = {
            "O": _observed_mean(observed, metric),
            "T": _replay_mean(replay_result["trace_reconstructed"], metric),
            "D": _replay_mean(replay_modes["device_only"]["metrics"], metric),
            "L": _replay_mean(
                replay_modes["device_plus_launch"]["metrics"], metric
            ),
            "S": _replay_mean(
                replay_modes[
                    "device_plus_launch_plus_observed_scheduler_cpu"
                ]["metrics"],
                metric,
            ),
            "F_D": dynamic_modes["device_only"]["metrics"][metric]["predicted"],
            "F_L": dynamic_modes["device_plus_launch"]["metrics"][metric][
                "predicted"
            ],
            "F_S": full_metrics[metric]["predicted"],
        }
        points_by_metric[metric] = decompose_metric(points)
    result = {
        "simulation_source_commit": source_commit,
        "cell_id": cell.cell_id,
        "hardware_id": cell.hardware.hardware_id,
        "model_id": cell.model.dataset_model_id,
        "case_id": cell.case_id,
        "target_qps": cell.qps,
        "request_count": len(case.requests),
        "configured_depth": configured_depth,
        "metrics": points_by_metric,
        "schedule_structure": {
            "observed": observed_structure,
            "dynamic_device_only": dynamic_modes["device_only"]["structure"],
            "dynamic_device_plus_launch": dynamic_modes[
                "device_plus_launch"
            ]["structure"],
            "dynamic_full": _full_structure(full_cell),
        },
        "input_sha256": {
            "manifest": _sha256(case_path / "manifest.json"),
            "requests": _sha256(case_path / "requests.csv"),
            "metrics": _sha256(case_path / "metrics.json"),
            "scheduler_trace": _sha256(original_trace),
        },
        "dynamic_mode_argv_sha256": {
            mode: dynamic_modes[mode]["forward_argv_sha256"]
            for mode in ("device_only", "device_plus_launch")
        },
        "replay_result": str(Path(replay_output_path)),
    }
    _write_json_atomic(Path(case_cache_path), result)
    return result


def _run_one_final_serving_quantile_case(
    repository_root: str,
    config_path: str,
    candidate_case_path: str,
    cell_id: str,
    host_profile_path: str,
    scheduler_host_profile_path: str,
    source_commit: str,
    load_validated_case: Callable[[Path], object],
    cache_path_value: str,
) -> dict[str, object]:
    root = Path(repository_root)
    cache_path = Path(cache_path_value)
    if cache_path.is_file():
        cached = _mapping(
            json.loads(cache_path.read_text()), "cached final Serving quantiles"
        )
        if (cached["schema_version"] != 3
                or cached["kind"] != "full_only_serving_report_cell"
                or cached["source_commit"] != source_commit
                or cached["cell_id"] != cell_id
                or cached["serving_timing_mode"]
                != "device_plus_launch_plus_scheduler"):
            raise ValueError(
                f"cached final Serving quantile identity differs: {cache_path}"
            )
        return dict(_mapping(cached["cell"], "cached final Serving cell"))

    config, cell, case, _observed, argv = _prepare_final_serving_quantile_case(
        repository_root,
        config_path,
        candidate_case_path,
        cell_id,
        host_profile_path,
        scheduler_host_profile_path,
        load_validated_case,
    )
    result = compact.evaluate_validated_case(
        root,
        config,
        cell,
        case,
        simulation_runner.run_cli_args,
        source_commit=source_commit,
        alignment_checked=False,
        execution_mode="accelerated",
        serving_timing_mode="device_plus_launch_plus_scheduler",
        host_launch_profile_path=host_profile_path,
        scheduler_host_profile_path=scheduler_host_profile_path,
    )
    configured_depth = compact.scheduler_pipeline_depth_from_validated_case(case)
    result = {
        **result,
        "depth_identity": {
            "configured_depth": configured_depth,
            "resolved_depth": configured_depth,
            "mode": "dataset_native",
            "evidence": case.manifest["runtime"][
                "scheduler_pipeline_depth_evidence"
            ],
        },
    }
    _write_json_atomic(cache_path, {
        "schema_version": 3,
        "kind": "full_only_serving_report_cell",
        "source_commit": source_commit,
        "cell_id": cell.cell_id,
        "serving_timing_mode": "device_plus_launch_plus_scheduler",
        "forward_argv_sha256": _canonical_sha256(argv),
        "cell": result,
    })
    gc.collect()
    return result


def _prepare_final_serving_quantile_case(
    repository_root: str,
    config_path: str,
    candidate_case_path: str,
    cell_id: str,
    host_profile_path: str,
    scheduler_host_profile_path: str,
    load_validated_case: Callable[[Path], object],
) -> tuple[object, object, object, Mapping[str, object], list[str]]:
    """Validate and prepare one case without starting its simulation."""

    root = Path(repository_root)
    config = compact.load_phase1_config(config_path)
    cell = compact.select_phase1_cells(config, (cell_id,), all_cases=False)[0]
    case = load_validated_case(Path(candidate_case_path))
    compact._validate_case_binding(config, cell, case)
    observed = compact._observed_metrics(case)
    argv = compact.build_simulation_argv(
        root,
        config,
        cell,
        case,
        execution_mode="accelerated",
        serving_timing_mode="device_plus_launch_plus_scheduler",
        host_launch_profile_path=host_profile_path,
        scheduler_host_profile_path=scheduler_host_profile_path,
    )
    if case.scheduler_steps is None:
        raise ValueError(f"{cell_id} requires observed Scheduler progress")
    return config, cell, case, observed, argv


def _select_configured_cells(all_configured, selected_cell_ids):
    """Return the configured cells the caller asked for, in request order.

    A requested id the configuration does not contain is refused rather than
    silently dropped; a duplicate request is refused because the result would
    be ambiguous.  With no explicit selection every configured cell is used.
    """

    if not selected_cell_ids:
        return all_configured
    requested = tuple(selected_cell_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("selected cell IDs contain duplicates")
    available = {cell.cell_id for _config, cell in all_configured}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"unknown selected cells: {unknown}")
    return tuple(
        (config, cell)
        for requested_id in requested
        for config, cell in all_configured
        if cell.cell_id == requested_id
    )


def run_scheduler_depth_breakdown(
    repository_root: str | Path,
    dataset_root: str | Path,
    full_results_path: str | Path,
    run_id: str,
    *,
    jobs: int,
    selected_cell_ids: Sequence[str] = (),
    resume: bool = False,
    load_validated_case: Callable[[Path], object],
) -> dict[str, object]:
    """Run exact fixed/dynamic attribution for selected or all official cases."""

    root = Path(repository_root).resolve()
    dataset_base = Path(dataset_root).resolve(strict=True)
    full_path = Path(full_results_path).resolve(strict=True)
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    source_commit = compact.repository_source_commit(root)
    full_payload = _mapping(json.loads(full_path.read_text()), "full results")
    full_cells = _full_cell_index(full_payload)
    all_configured = tuple(
        (config, cell)
        for config, cells in compact.official_metrics_suite_runs(root)
        for cell in cells
    )
    selected = _select_configured_cells(all_configured, selected_cell_ids)
    profiles = compact.resolve_host_launch_profile_paths(
        root, (), {cell.hardware.hardware_id for _config, cell in selected}
    )
    if profiles is None:
        raise ValueError("Scheduler breakdown requires Host launch profiles")
    scheduler_profiles = compact.resolve_scheduler_host_profile_paths(
        root, (), {cell.hardware.hardware_id for _config, cell in selected}
    )
    if scheduler_profiles is None:
        raise ValueError("Scheduler breakdown requires Scheduler Host profiles")

    output_root = root / "experiment_logs/serving/scheduler_depth_breakdown"
    destination = (output_root / run_id).resolve()
    if output_root.resolve() not in destination.parents:
        raise ValueError("breakdown output must remain in experiment_logs")
    if not isinstance(resume, bool):
        raise TypeError("resume must be boolean")
    destination.mkdir(parents=True, exist_ok=resume)
    replay_dir = destination / "replay"
    replay_dir.mkdir(exist_ok=resume)
    dynamic_cache_dir = destination / "dynamic_modes"
    dynamic_cache_dir.mkdir(exist_ok=resume)
    final_quantile_dir = destination / "final_serving_quantiles"
    final_quantile_dir.mkdir(exist_ok=resume)
    case_cache_dir = destination / "case_results"
    case_cache_dir.mkdir(exist_ok=resume)
    jobspecs = []
    final_quantile_jobspecs = []
    for config, cell in selected:
        case_path = (
            dataset_base
            / cell.hardware.hardware_id
            / cell.model.dataset_model_id
            / cell.case_id
        ).resolve(strict=True)
        replay_path = replay_dir / f"{cell.cell_id.replace('/', '__')}.json"
        case_cache_path = (
            case_cache_dir / f"{cell.cell_id.replace('/', '__')}.json"
        )
        final_quantile_cache_path = (
            final_quantile_dir / f"{cell.cell_id.replace('/', '__')}.json"
        )
        jobspecs.append((
            str(root),
            str(config.source_path),
            str(case_path),
            cell.cell_id,
            str(profiles[cell.hardware.hardware_id]),
            source_commit,
            full_cells[cell.cell_id],
            str(replay_path),
            str(dynamic_cache_dir),
            load_validated_case,
            str(case_cache_path),
        ))
        final_quantile_jobspecs.append((
            str(root),
            str(config.source_path),
            str(case_path),
            cell.cell_id,
            str(profiles[cell.hardware.hardware_id]),
            str(scheduler_profiles[cell.hardware.hardware_id]),
            source_commit,
            load_validated_case,
            str(final_quantile_cache_path),
        ))
    started = time.perf_counter()
    cases = _run_parallel(jobspecs, jobs)
    payload = {
        "schema_version": 1,
        "kind": "scheduler_depth_ttft_tpot_signed_breakdown",
        "diagnostic_only": True,
        "source_commit": source_commit,
        "dataset_root": str(dataset_base),
        "full_results": {
            "path": str(full_path),
            "sha256": _sha256(full_path),
            "declared_source_commit": full_payload["source_commit"],
        },
        "case_count": len(cases),
        "metric_row_count": len(cases) * len(_METRICS),
        "evaluation_wall_time_seconds": time.perf_counter() - started,
        "decomposition_order": list(_COMPONENTS),
        "cases": cases,
        "hardware_summary": _hardware_summary(cases),
    }
    paths = _write_outputs(destination, payload)
    diagnostic_plot_paths = write_scheduler_depth_diagnostic_plots(
        destination / "diagnostic_plots", cases, full_cells
    )
    paths.update({
        f"diagnostic_plot_{name}": path
        for name, path in diagnostic_plot_paths.items()
    })
    replay_cells = [
        json.loads(path.read_text()) for path in sorted(replay_dir.glob("*.json"))
    ]
    plot_dir = destination / "plots"
    if plot_dir.exists():
        if not resume:
            raise ValueError("plot directory already exists without resume")
        plot_paths = {
            path.stem: path
            for path in sorted(plot_dir.glob("*scheduling_trajectory_replay*.png"))
        }
        if len(plot_paths) != 6:
            raise ValueError("resumed plot directory must contain six PNG files")
    else:
        plot_paths = write_scheduling_trajectory_replay_quantile_plots(
            plot_dir, replay_cells
        )
    paths.update({f"plot_{name}": path for name, path in plot_paths.items()})
    final_quantile_cells = _run_final_serving_quantile_parallel(
        final_quantile_jobspecs, jobs
    )
    final_plot_paths = sharegpt_report.write_sharegpt_latency_quantile_plots(
        plot_dir,
        final_quantile_cells,
        filename_prefix="final_serving",
    )
    paths.update({
        f"plot_{name}": path for name, path in final_plot_paths.items()
    })
    hashes = {
        str(path.relative_to(destination)): _sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    }
    hash_path = destination / "hashes.json"
    hash_path.write_text(json.dumps(
        {"schema_version": 1, "sha256": hashes}, indent=2, sort_keys=True
    ) + "\n")
    paths["hashes"] = hash_path
    return {
        "run_directory": str(destination.relative_to(root)),
        "case_count": len(cases),
        "metric_row_count": len(cases) * len(_METRICS),
        "artifacts": {
            name: str(path.relative_to(root)) for name, path in paths.items()
        },
    }


def run_scheduler_depth_final_serving_quantiles(
    repository_root: str | Path,
    dataset_root: str | Path,
    full_results_path: str | Path,
    run_id: str,
    *,
    jobs: int,
    selected_cell_ids: Sequence[str] = (),
    load_validated_case: Callable[[Path], object],
) -> dict[str, object]:
    """Generate final dynamic Serving TTFT/TPOT quantile plots for a report."""

    root = Path(repository_root).resolve()
    dataset_base = Path(dataset_root).resolve(strict=True)
    full_path = Path(full_results_path).resolve(strict=True)
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    source_commit = compact.repository_source_commit(root)
    full_payload = _mapping(json.loads(full_path.read_text()), "full results")
    full_cells = _full_cell_index(full_payload)
    all_configured = tuple(
        (config, cell)
        for config, cells in compact.official_metrics_suite_runs(root)
        for cell in cells
        if cell.cell_id in full_cells
    )
    selected = _select_configured_cells(all_configured, selected_cell_ids)
    profiles = compact.resolve_host_launch_profile_paths(
        root, (), {cell.hardware.hardware_id for _config, cell in selected}
    )
    if profiles is None:
        raise ValueError("final Serving quantiles require Host launch profiles")
    scheduler_profiles = compact.resolve_scheduler_host_profile_paths(
        root, (), {cell.hardware.hardware_id for _config, cell in selected}
    )
    if scheduler_profiles is None:
        raise ValueError(
            "final Serving quantiles require Scheduler Host profiles")

    output_root = root / "experiment_logs/serving/scheduler_depth_breakdown"
    destination = (output_root / run_id).resolve()
    if output_root.resolve() not in destination.parents:
        raise ValueError("breakdown output must remain in experiment_logs")
    if not destination.is_dir():
        raise ValueError(f"breakdown run directory does not exist: {destination}")
    breakdown_path = destination / "breakdown.json"
    if not breakdown_path.is_file():
        raise ValueError("breakdown run directory is missing breakdown.json")

    final_quantile_dir = destination / "final_serving_quantiles"
    final_quantile_dir.mkdir(exist_ok=True)
    jobspecs = []
    for config, cell in selected:
        case_path = (
            dataset_base
            / cell.hardware.hardware_id
            / cell.model.dataset_model_id
            / cell.case_id
        ).resolve(strict=True)
        final_quantile_cache_path = (
            final_quantile_dir / f"{cell.cell_id.replace('/', '__')}.json"
        )
        jobspecs.append((
            str(root),
            str(config.source_path),
            str(case_path),
            cell.cell_id,
            str(profiles[cell.hardware.hardware_id]),
            str(scheduler_profiles[cell.hardware.hardware_id]),
            source_commit,
            load_validated_case,
            str(final_quantile_cache_path),
        ))
    cells = _run_final_serving_quantile_parallel(jobspecs, jobs)
    plot_paths = sharegpt_report.write_sharegpt_latency_quantile_plots(
        destination / "plots",
        cells,
        filename_prefix="final_serving",
    )
    payload = _mapping(json.loads(breakdown_path.read_text()), "breakdown payload")
    (destination / "report.md").write_text(_render_report(payload))
    hashes = {
        str(path.relative_to(destination)): _sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    }
    hash_path = destination / "hashes.json"
    hash_path.write_text(json.dumps(
        {"schema_version": 1, "sha256": hashes}, indent=2, sort_keys=True
    ) + "\n")
    return {
        "run_directory": str(destination.relative_to(root)),
        "case_count": len(cells),
        "plot_count": len(plot_paths),
        "artifacts": {
            name: str(path.relative_to(root)) for name, path in plot_paths.items()
        } | {"hashes": str(hash_path.relative_to(root))},
    }


def run_full_only_serving_evaluation(
    repository_root: str | Path,
    dataset_root: str | Path,
    run_id: str,
    *,
    jobs: int,
    selected_cell_ids: Sequence[str] = (),
    reuse_results_paths: Sequence[str | Path] = (),
    load_validated_case: Callable[[Path], object],
    nvidia_h200_vllm_version: str | None = None,
    nvidia_h200_dataset_root: str | Path | None = None,
) -> dict[str, object]:
    """Run selected complete-timing cases, optionally reusing exact cell results.

    ``nvidia_h200_vllm_version`` adds the 15 H200 cells of that captured
    vLLM version, read from ``dataset_root`` (the canonical cases are vLLM
    0.30.0) or, when given, from the ``nvidia_h200_dataset_root`` candidate
    root; every other hardware keeps ``dataset_root``.
    """

    root = Path(repository_root).resolve()
    dataset_base = Path(dataset_root).resolve(strict=True)
    if nvidia_h200_dataset_root is not None and nvidia_h200_vllm_version is None:
        raise ValueError("an H200 dataset root requires its H200 vLLM version")
    dataset_bases = {}
    if nvidia_h200_dataset_root is not None:
        dataset_bases["nvidia_h200"] = Path(
            nvidia_h200_dataset_root).resolve(strict=True)
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    source_commit = compact.repository_source_commit(root)
    configured = tuple(
        (config, cell)
        for config, cells in compact.official_metrics_suite_runs(
            root, nvidia_h200_vllm_version=nvidia_h200_vllm_version)
        for cell in cells
    )
    if selected_cell_ids:
        requested = tuple(selected_cell_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("selected cell IDs contain duplicates")
        available = {cell.cell_id for _config, cell in configured}
        unknown = sorted(set(requested) - available)
        if unknown:
            raise ValueError(f"unknown selected cells: {unknown}")
        selected = tuple(
            (config, cell)
            for requested_id in requested
            for config, cell in configured
            if cell.cell_id == requested_id
        )
    else:
        selected = configured
    reused_cells, reuse_sources = _load_reusable_full_only_cells(
        reuse_results_paths,
        selected,
        source_commit,
    )
    pending = tuple(
        (config, cell)
        for config, cell in selected
        if cell.cell_id not in reused_cells
    )
    _preflight_launch_artifacts({
        cell.hardware.hardware_id for _config, cell in pending
    })
    versioned_hardware = compact.host_profile_variant_hardware()
    published_hardware = {
        cell.hardware.hardware_id for _config, cell in pending
    } - versioned_hardware
    profiles = compact.resolve_host_launch_profile_paths(
        root, (), published_hardware)
    scheduler_profiles = compact.resolve_scheduler_host_profile_paths(
        root, (), published_hardware)
    if published_hardware and (profiles is None or scheduler_profiles is None):
        raise ValueError(
            "full-only Serving evaluation requires Host launch and Scheduler "
            "Host profiles")
    output_root = root / "experiment_logs/serving/sharegpt_full"
    destination = (output_root / run_id).resolve()
    if output_root.resolve() not in destination.parents:
        raise ValueError("full-only Serving output must remain in experiment_logs")
    cell_dir = destination / "cells"
    jobspecs = []
    for config, cell in pending:
        hardware_id = cell.hardware.hardware_id
        case_path = (
            (dataset_bases[hardware_id] if hardware_id in dataset_bases
             else dataset_base)
            / hardware_id
            / cell.model.dataset_model_id
            / cell.case_id
        ).resolve(strict=True)
        if hardware_id in versioned_hardware:
            # The case manifest is validated against the suite's pinned
            # SHA256 before any worker starts; its runtime selects profiles.
            case_runtime = _mapping(
                json.loads((case_path / "manifest.json").read_text())["runtime"],
                "case manifest runtime")
            if case_runtime["runtime_version"] != nvidia_h200_vllm_version:
                raise ValueError(
                    f"{cell.cell_id} was captured on vLLM "
                    f"{case_runtime['runtime_version']!r}, not the selected "
                    f"{nvidia_h200_vllm_version!r}")
            variant = compact.case_host_profile_paths(
                root, hardware_id, str(case_runtime["runtime_version"]))
            launch_profile = variant["host_launch"]
            scheduler_profile = variant["scheduler_host"]
        else:
            launch_profile = profiles[hardware_id]
            scheduler_profile = scheduler_profiles[hardware_id]
        jobspecs.append((
            str(root),
            str(config.source_path),
            str(case_path),
            cell.cell_id,
            str(launch_profile),
            str(scheduler_profile),
            source_commit,
            load_validated_case,
            str(cell_dir / f"{cell.cell_id.replace('/', '__')}.json"),
        ))
    _preflight_final_serving_quantile_cases(jobspecs)
    destination.mkdir(parents=True, exist_ok=False)
    cell_dir.mkdir()
    started = time.perf_counter()
    evaluated_cells = _run_final_serving_quantile_parallel(jobspecs, jobs)
    evaluation_wall = time.perf_counter() - started
    cells_by_id = dict(reused_cells)
    cells_by_id.update({cell["cell_id"]: cell for cell in evaluated_cells})
    cells = [cells_by_id[cell.cell_id] for _config, cell in selected]
    payload = {
        "schema_version": 1,
        "kind": "sharegpt_full_only_standard_report_input",
        "source_commit": source_commit,
        "dataset_root": str(dataset_base),
        "hardware_dataset_roots": {
            hardware_id: str(path) for hardware_id, path in dataset_bases.items()
        },
        "nvidia_h200_vllm_version": nvidia_h200_vllm_version,
        "serving_timing_mode": "device_plus_launch_plus_scheduler",
        "cell_count": len(cells),
        "evaluation_wall_time_seconds": evaluation_wall,
        "reused_cell_count": len(reused_cells),
        "newly_evaluated_cell_count": len(evaluated_cells),
        "reused_result_paths": reuse_sources,
        "cells": cells,
    }
    result_path = destination / "results.json"
    _write_json_atomic(result_path, payload)
    report_paths = sharegpt_report.write_sharegpt_evaluation_report(
        destination,
        cells,
        run_elapsed_wall_time_seconds=evaluation_wall,
    )
    return {
        "run_directory": str(destination.relative_to(root)),
        "result_path": str(result_path.relative_to(root)),
        "source_commit": source_commit,
        "serving_timing_mode": "device_plus_launch_plus_scheduler",
        "case_count": len(cells),
        "evaluation_wall_time_seconds": evaluation_wall,
        "reused_cell_count": len(reused_cells),
        "newly_evaluated_cell_count": len(evaluated_cells),
        "artifacts": {
            name: str(path.relative_to(root))
            for name, path in report_paths.items()
        },
    }


def _load_reusable_full_only_cells(
    reuse_results_paths: Sequence[str | Path],
    selected: Sequence[tuple[object, object]],
    source_commit: str,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Load completed cells only when they exactly match this full-serving run."""

    configured = {
        cell.cell_id: cell
        for _config, cell in selected
    }
    reused = {}
    sources = []
    for raw_path in reuse_results_paths:
        path = Path(raw_path).resolve(strict=True)
        payload = _mapping(json.loads(path.read_text()), "reused Serving results")
        if payload["schema_version"] != 1:
            raise ValueError("reused Serving results schema_version must equal 1")
        if payload["source_commit"] != source_commit:
            raise ValueError("reused Serving results source commit differs")
        raw_cells = _sequence(payload["cells"], "reused Serving cells")
        source_ids = set()
        for raw_cell in raw_cells:
            reusable = dict(_mapping(raw_cell, "reused Serving cell"))
            cell_id = reusable["cell_id"]
            if not isinstance(cell_id, str):
                raise TypeError("reused Serving cell ID must be text")
            if cell_id in source_ids:
                raise ValueError(f"duplicate reused Serving cell: {cell_id}")
            source_ids.add(cell_id)
            if cell_id not in configured:
                continue
            if cell_id in reused:
                raise ValueError(f"reused Serving cell appears in multiple inputs: {cell_id}")
            _validate_reusable_full_only_cell(reusable, configured[cell_id])
            reused[cell_id] = reusable
        sources.append(str(path))
    return reused, sources


def _validate_reusable_full_only_cell(
    reusable: Mapping[str, object],
    configured_cell: object,
) -> None:
    """Reject a reused cell unless workload identity and timing contract agree."""

    expected = configured_cell
    identity = {
        "hardware_id": expected.hardware.hardware_id,
        "model_id": expected.model.dataset_model_id,
        "case_id": expected.case_id,
        "target_qps": expected.qps,
        "manifest_sha256": expected.manifest_sha256,
        "requests_sha256": expected.requests_sha256,
    }
    for field, value in identity.items():
        if reusable[field] != value:
            raise ValueError(f"reused Serving {field} differs for {expected.cell_id}")
    if reusable["prediction_status"] != "completed":
        raise ValueError(f"reused Serving cell is not completed: {expected.cell_id}")
    if reusable["execution_mode"] != "accelerated":
        raise ValueError(f"reused Serving execution mode differs: {expected.cell_id}")
    if reusable["serving_timing_mode"] != "device_plus_launch_plus_scheduler":
        raise ValueError(f"reused Serving timing mode differs: {expected.cell_id}")
    _mapping(reusable["metrics"], "reused Serving metrics")
    _mapping(reusable["host_timing"], "reused Host timing")


def _preflight_final_serving_quantile_cases(
    jobspecs: Sequence[tuple[object, ...]],
) -> None:
    """Validate every case synchronously before starting expensive workers."""

    for spec in jobspecs:
        _prepare_final_serving_quantile_case(
            str(spec[0]),
            str(spec[1]),
            str(spec[2]),
            str(spec[3]),
            str(spec[4]),
            str(spec[5]),
            spec[7],
        )


def _preflight_launch_artifacts(hardware_ids: set[str]) -> None:
    """Validate canonical launch artifacts before starting case workers."""

    unknown = hardware_ids - set(_GEMM_LAUNCH_SELECTION_SOURCE)
    if unknown:
        raise ValueError(f"unknown launch preflight hardware: {sorted(unknown)}")
    for hardware_id in sorted(hardware_ids):
        source_id = _GEMM_LAUNCH_SELECTION_SOURCE[hardware_id]
        selection = load_gemm_launch_selection(
            source_id,
            software_revision=gemm_launch_software_revision(source_id),
        )
        if not selection:
            raise ValueError(
                f"{hardware_id} canonical GEMM launch selection is empty"
            )


def _run_parallel(jobspecs: Sequence[tuple[object, ...]], jobs: int):
    results = {}
    pending = {}
    for index, spec in enumerate(jobspecs):
        case_cache_path = Path(str(spec[-1]))
        if case_cache_path.is_file():
            cached = _mapping(
                json.loads(case_cache_path.read_text()), "cached case result"
            )
            if cached["cell_id"] != spec[3]:
                raise ValueError("cached case result identity differs")
            if cached["simulation_source_commit"] != spec[5]:
                raise ValueError("cached case result source commit differs")
            results[index] = cached
        else:
            pending[index] = spec
    if not pending:
        return [results[index] for index in range(len(jobspecs))]
    with ProcessPoolExecutor(max_workers=min(jobs, len(pending))) as executor:
        futures = {
            executor.submit(_run_one_case, *spec): index
            for index, spec in pending.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[index] for index in range(len(jobspecs))]


def _run_final_serving_quantile_parallel(
    jobspecs: Sequence[tuple[object, ...]], jobs: int
) -> list[dict[str, object]]:
    results = {}
    pending = {}
    for index, spec in enumerate(jobspecs):
        cache_path = Path(str(spec[-1]))
        if cache_path.is_file():
            cached = _mapping(
                json.loads(cache_path.read_text()), "cached final Serving quantiles"
            )
            if cached["cell_id"] != spec[3]:
                raise ValueError("cached final Serving quantile identity differs")
            if cached["source_commit"] != spec[6]:
                raise ValueError("cached final Serving quantile source differs")
            results[index] = dict(
                _mapping(cached["cell"], "cached final Serving cell")
            )
        else:
            pending[index] = spec
    if pending:
        with ProcessPoolExecutor(max_workers=min(jobs, len(pending))) as executor:
            futures = {
                executor.submit(_run_one_final_serving_quantile_case, *spec): index
                for index, spec in pending.items()
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    return [results[index] for index in range(len(jobspecs))]




def _project_v5_scheduler_trace(
    path: Path,
    destination,
    requests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    phase_counts = {phase: 0 for phase in replay.PHASES}
    scheduled_tokens = 0
    scheduled_requests = 0
    device_queue_wait_ns = 0
    completion_queue_wait_ns = 0
    step_count = 0
    scheduler_schema_version = None
    decode_by_request = [0] * len(requests)
    if path.suffix == ".zst":
        try:
            trace_bytes = zstandard.ZstdDecompressor().decompress(path.read_bytes())
        except zstandard.ZstdError as error:
            raise ValueError(f"Scheduler trace is invalid zstd: {path}") from error
        try:
            trace_text = trace_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Scheduler trace must be UTF-8: {path}") from error
    else:
        trace_text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(trace_text.splitlines(), start=1):
            row = _mapping(json.loads(line), f"Scheduler row {line_number}")
            common = (
                replay.SCHEDULER_FIELDS
                | replay.SCHEDULER_RUNTIME_FIELDS
                | {"schema_version"}
            )
            valid_layouts = (
                (common | replay.SCHEDULER_RANK_OBSERVATION_FIELDS, 4),
                (common | replay.SCHEDULER_RANK_OBSERVATION_FIELDS, 5),
                (common | replay.SCHEDULER_FLAT_GRAPH_FIELDS, 3),
            )
            matching_versions = [
                version for fields, version in valid_layouts if set(row) == fields
            ]
            if not matching_versions:
                raise ValueError(
                    f"candidate Scheduler row fields differ at {line_number}"
                )
            if row["schema_version"] not in matching_versions:
                raise ValueError(
                    "candidate Scheduler row version disagrees with its layout"
                )
            if scheduler_schema_version is None:
                scheduler_schema_version = row["schema_version"]
            elif scheduler_schema_version != row["schema_version"]:
                raise ValueError("candidate Scheduler schemas vary across rows")
            destination.write(json.dumps(row, sort_keys=True) + "\n")
            prefill = _integer(row["prefill_tokens"], "prefill tokens")
            decode = _integer(row["decode_tokens"], "decode tokens")
            phase = "mixed" if prefill and decode else (
                "prefill" if prefill else "decode"
            )
            phase_counts[phase] += 1
            scheduled_tokens += _integer(row["scheduled_tokens"], "scheduled tokens")
            scheduled_requests += _integer(
                row["scheduled_request_count"], "scheduled requests"
            )
            for raw_request in _sequence(
                row["scheduled_requests"], "scheduled request rows"
            ):
                scheduled_request = _mapping(
                    raw_request, "scheduled request row"
                )
                request_idx = _integer(
                    scheduled_request["request_idx"], "request index"
                )
                if request_idx >= len(requests):
                    raise ValueError("scheduled request index is out of range")
                decode_by_request[request_idx] += _integer(
                    scheduled_request["decode_tokens"], "request decode tokens"
                )
            device_queue_wait_ns += max(
                0,
                _integer(row["device_execution_start_ns"], "device start")
                - _integer(row["schedule_end_ns"], "schedule end"),
            )
            completion_queue_wait_ns += max(
                0,
                _integer(row["scheduler_update_start_ns"], "update start")
                - _integer(row["device_execution_end_ns"], "device end"),
            )
            step_count += 1
    if step_count == 0:
        raise ValueError("candidate Scheduler trace is empty")
    offsets = {
        decode_by_request[index]
        - (_integer(request["output_tokens"], "request output tokens") - 1)
        for index, request in enumerate(requests)
    }
    if offsets not in ({0}, {1}):
        raise ValueError(
            f"Scheduler decode-work offsets are not uniformly zero or one: {offsets}"
        )
    decode_work_offset = next(iter(offsets))
    return {
        "scheduler_schema_version": scheduler_schema_version,
        "step_count": step_count,
        "scheduled_tokens": scheduled_tokens,
        "scheduled_requests": scheduled_requests,
        "mean_scheduled_tokens_per_step": scheduled_tokens / step_count,
        "mean_scheduled_requests_per_step": scheduled_requests / step_count,
        "phase_step_count": phase_counts,
        "phase_step_fraction": {
            phase: count / step_count for phase, count in phase_counts.items()
        },
        "device_queue_wait_s": device_queue_wait_ns / 1_000_000_000.0,
        "completion_to_update_wait_s": (
            completion_queue_wait_ns / 1_000_000_000.0
        ),
        "decode_work_token_offset": decode_work_offset,
        "decode_work_semantics": (
            "includes_one_terminal_token_per_request"
            if decode_work_offset == 1
            else "client_output_tokens_minus_first_token"
        ),
    }


def _restore_client_visible_tpot(
    replay_result: Mapping[str, object],
    requests: Sequence[Mapping[str, object]],
) -> None:
    metric_sets = [replay_result["trace_reconstructed"]]
    modes = replay_result["scheduling_trajectory_replay_modes"]
    metric_sets.extend(modes[mode]["metrics"] for mode in (
        "device_only",
        "device_plus_launch",
        "device_plus_launch_plus_observed_scheduler_cpu",
    ))
    for metrics in metric_sets:
        rows = _sequence(metrics["per_request"], "replay request metrics")
        if len(rows) != len(requests):
            raise ValueError("replay request count differs during TPOT normalization")
        values = []
        for index, raw_row in enumerate(rows):
            row = _mapping(raw_row, "replay request metric")
            if row["request_idx"] != index:
                raise ValueError("replay request order differs during TPOT normalization")
            denominator = _integer(
                requests[index]["output_tokens"], "client output tokens"
            ) - 1
            if denominator <= 0:
                raise ValueError("TPOT requires at least two client-visible tokens")
            tpot = (row["e2e_latency_ms"] - row["ttft_ms"]) / denominator
            row["tpot_ms"] = tpot
            values.append(tpot)
        metrics["latency"]["tpot_ms"] = replay._summarize_values(values)
    trace_metrics = replay_result["trace_reconstructed"]
    for mode in (
        "device_only",
        "device_plus_launch",
        "device_plus_launch_plus_observed_scheduler_cpu",
    ):
        modes[mode]["comparison_to_trace_reconstructed"] = (
            replay.compare_request_metrics(trace_metrics, modes[mode]["metrics"])
        )
    replay_result["contract"]["client_visible_tpot_denominator_restored"] = True


def _dynamic_structure(
    host_timing: Mapping[str, object], metrics: Mapping[str, object]
) -> dict[str, object]:
    aggregate = _mapping(host_timing["aggregate"], "Host aggregate")
    steps = _sequence(host_timing["steps"], "Host steps")
    phase_counts = {phase: 0 for phase in replay.PHASES}
    tokens = 0
    requests = 0
    for raw_step in steps:
        step = _mapping(raw_step, "Host step")
        phase = step["phase"]
        if phase not in phase_counts:
            raise ValueError(f"unexpected dynamic phase: {phase!r}")
        phase_counts[phase] += 1
        tokens += _integer(step["prefill_token_count"], "prefill token count")
        tokens += _integer(step["decode_token_count"], "decode token count")
        requests += _integer(
            step["scheduled_request_count"], "scheduled request count"
        )
    step_count = _integer(aggregate["step_count"], "step count")
    critical_path = _mapping(metrics["critical_path"], "critical path")
    critical_steps = _sequence(critical_path["steps"], "critical path steps")
    queue = {
        "device_queue_wait_s": 0.0,
        "completion_to_update_wait_s": 0.0,
    }
    for raw_step in critical_steps:
        step = _mapping(raw_step, "critical path step")
        contributions = _mapping(
            step["contribution_s_by_category"], "step contributions"
        )
        if "scheduler_device_queue_wait" in contributions:
            queue["device_queue_wait_s"] += _finite(
                contributions["scheduler_device_queue_wait"],
                "device queue wait",
            )
        if "scheduler_completion_queue_wait" in contributions:
            queue["completion_to_update_wait_s"] += _finite(
                contributions["scheduler_completion_queue_wait"],
                "completion queue wait",
            )
    return {
        "step_count": step_count,
        "scheduled_tokens": tokens,
        "scheduled_requests": requests,
        "mean_scheduled_tokens_per_step": tokens / step_count,
        "mean_scheduled_requests_per_step": requests / step_count,
        "phase_step_count": phase_counts,
        "phase_step_fraction": {
            phase: count / step_count for phase, count in phase_counts.items()
        },
        "predicted_device_s": aggregate["predicted_device_s"],
        "launch_cpu_s": aggregate["launch_cpu_s"],
        "scheduler_cpu_s": aggregate["scheduler_cpu_s"],
        "schedule_makespan_s": aggregate["schedule_makespan_s"],
        **queue,
    }


def _full_structure(cell: Mapping[str, object]) -> dict[str, object]:
    host_timing = _mapping(cell["host_timing"], "full Host timing")
    aggregate = _mapping(host_timing["aggregate"], "full Host aggregate")
    return {
        field: aggregate[field]
        for field in (
            "step_count",
            "predicted_device_s",
            "launch_cpu_s",
            "scheduler_cpu_s",
            "schedule_makespan_s",
        )
    }


def _validated_full_cell(
    value: Mapping[str, object], cell_id: str, observed: Mapping[str, object]
) -> Mapping[str, object]:
    cell = _mapping(value, "full cell")
    if cell["cell_id"] != cell_id:
        raise ValueError("full cell identity differs")
    if cell["prediction_status"] != "completed":
        raise ValueError(f"full cell is not completed: {cell_id}")
    depth = _mapping(cell["depth_identity"], "full depth identity")
    if depth["mode"] != "dataset_native" or depth["resolved_depth"] != 2:
        raise ValueError(f"full cell is not dataset-native depth=2: {cell_id}")
    if cell["serving_timing_mode"] != "device_plus_launch_plus_scheduler":
        raise ValueError("full cell does not use complete Host timing")
    metrics = _mapping(cell["metrics"], "full metrics")
    if all(metric in metrics for metric in _METRICS):
        metric_comparisons = metrics
    elif "latency" in metrics:
        metric_comparisons = _mapping(metrics["latency"], "full latency metrics")
        missing = set(_METRICS) - set(metric_comparisons)
        if missing:
            raise ValueError(
                f"full latency metrics are missing required entries: {sorted(missing)}"
            )
    else:
        raise ValueError("full metrics contain no supported latency comparison")
    normalized = {}
    for metric in _METRICS:
        comparison = _mapping(metric_comparisons[metric], f"full {metric}")
        if {"observed", "predicted"} <= set(comparison):
            point = comparison
        elif "mean" in comparison:
            point = _mapping(comparison["mean"], f"full {metric} mean")
        else:
            raise ValueError(f"full {metric} has no supported comparison point")
        expected = _observed_mean(observed, metric)
        actual = _finite(point["observed"], f"full observed {metric}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"full and candidate observed {metric} differ")
        normalized[metric] = {
            "observed": actual,
            "predicted": _finite(point["predicted"], f"full predicted {metric}"),
        }
    return normalized


def _full_cell_index(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    cells = _sequence(payload["cells"], "full result cells")
    index = {}
    for value in cells:
        cell = _mapping(value, "full result cell")
        depth = _mapping(cell["depth_identity"], "depth identity")
        if depth["mode"] != "dataset_native":
            continue
        cell_id = cell["cell_id"]
        if not isinstance(cell_id, str):
            raise TypeError("full cell ID must be a string")
        if cell_id in index:
            raise ValueError(f"duplicate native full cell: {cell_id}")
        index[cell_id] = cell
    return index


def _hardware_summary(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summaries = []
    present_hardware = {str(case["hardware_id"]) for case in cases}
    unknown_hardware = present_hardware - set(HARDWARE_ORDER)
    if unknown_hardware:
        raise ValueError(f"unsupported hardware IDs: {sorted(unknown_hardware)}")
    hardware_ids = tuple(
        hardware_id
        for hardware_id in HARDWARE_ORDER
        if hardware_id in present_hardware
    )
    for hardware_id in hardware_ids:
        hardware_cases = [case for case in cases if case["hardware_id"] == hardware_id]
        for metric in _METRICS:
            rows = [case["metrics"][metric] for case in hardware_cases]
            normalized = {
                component: [
                    float(row["component_over_observed"][component]) for row in rows
                ]
                for component in _COMPONENTS
            }
            point_values = {
                point: [float(row["points"][point]) for row in rows]
                for point in _POINTS
            }
            final_ratios = [
                float(row["final_prediction_over_observed"]) for row in rows
            ]
            aggregate = {
                "hardware_id": hardware_id,
                "metric": metric,
                "case_count": len(rows),
                "macro_final_prediction_over_observed": fmean(final_ratios),
                "macro_final_signed_relative_error": fmean(final_ratios) - 1.0,
                "macro_component_over_observed": {
                    component: fmean(values)
                    for component, values in normalized.items()
                },
                "median_component_over_observed": {
                    component: median(values)
                    for component, values in normalized.items()
                },
                "p90_absolute_component_over_observed": {
                    component: _percentile(
                        sorted(abs(value) for value in values), 90
                    )
                    for component, values in normalized.items()
                },
                "weighted_final_prediction_over_observed": (
                    sum(point_values["F_S"]) / sum(point_values["O"])
                ),
            }
            closure = (
                1.0
                + sum(aggregate["macro_component_over_observed"].values())
                - aggregate["macro_final_prediction_over_observed"]
            )
            if abs(closure) > 1e-9:
                raise ValueError("hardware macro attribution does not close")
            aggregate["macro_closure_residual"] = closure
            summaries.append(aggregate)
    return summaries


def _write_outputs(destination: Path, payload: Mapping[str, object]):
    json_path = destination / "breakdown.json"
    per_case_path = destination / "per_case_breakdown.csv"
    summary_path = destination / "hardware_summary.csv"
    structure_path = destination / "schedule_structure.csv"
    report_path = destination / "report.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _write_per_case_csv(per_case_path, payload["cases"])
    _write_summary_csv(summary_path, payload["hardware_summary"])
    _write_structure_csv(structure_path, payload["cases"])
    report_path.write_text(_render_report(payload))
    return {
        "json": json_path,
        "per_case_csv": per_case_path,
        "hardware_summary_csv": summary_path,
        "schedule_structure_csv": structure_path,
        "markdown": report_path,
    }


def _write_per_case_csv(path: Path, cases: object) -> None:
    fields = [
        "cell_id", "hardware_id", "model_id", "case_id", "target_qps",
        "metric", *_POINTS,
    ]
    fields.extend(f"delta_{name}" for name in _COMPONENTS)
    fields.extend(f"delta_over_observed_{name}" for name in _COMPONENTS)
    fields.extend(f"share_of_final_error_{name}" for name in _COMPONENTS)
    fields.extend((
        "final_signed_error", "final_signed_relative_error",
        "final_absolute_percentage_error", "final_prediction_over_observed",
        "closure_residual", "closure_tolerance",
    ))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in _sequence(cases, "cases"):
            for metric in _METRICS:
                result = case["metrics"][metric]
                row = {
                    field: case[field]
                    for field in (
                        "cell_id", "hardware_id", "model_id", "case_id",
                        "target_qps",
                    )
                }
                row["metric"] = metric
                row.update(result["points"])
                row.update({
                    f"delta_{name}": result["components"][name]
                    for name in _COMPONENTS
                })
                row.update({
                    f"delta_over_observed_{name}": (
                        result["component_over_observed"][name]
                    )
                    for name in _COMPONENTS
                })
                row.update({
                    f"share_of_final_error_{name}": (
                        result["component_share_of_final_signed_error"][name]
                    )
                    for name in _COMPONENTS
                })
                for field in fields[-6:]:
                    row[field] = result[field]
                writer.writerow(row)


def _write_summary_csv(path: Path, summaries: object) -> None:
    fields = [
        "hardware_id", "metric", "case_count",
        "macro_final_prediction_over_observed",
        "macro_final_signed_relative_error",
        "weighted_final_prediction_over_observed", "macro_closure_residual",
    ]
    for prefix in (
        "macro_component_over_observed",
        "median_component_over_observed",
        "p90_absolute_component_over_observed",
    ):
        fields.extend(f"{prefix}_{name}" for name in _COMPONENTS)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in _sequence(summaries, "summaries"):
            row = {field: summary[field] for field in fields[:7]}
            for prefix in (
                "macro_component_over_observed",
                "median_component_over_observed",
                "p90_absolute_component_over_observed",
            ):
                row.update({
                    f"{prefix}_{name}": summary[prefix][name]
                    for name in _COMPONENTS
                })
            writer.writerow(row)


def _write_structure_csv(path: Path, cases: object) -> None:
    modes = (
        "observed", "dynamic_device_only", "dynamic_device_plus_launch",
        "dynamic_full",
    )
    fields = (
        "cell_id", "hardware_id", "model_id", "case_id", "target_qps", "mode",
        "step_count", "scheduled_tokens", "scheduled_requests",
        "mean_scheduled_tokens_per_step", "mean_scheduled_requests_per_step",
        "prefill_step_fraction", "mixed_step_fraction", "decode_step_fraction",
        "device_queue_wait_s", "completion_to_update_wait_s",
        "predicted_device_s", "launch_cpu_s", "scheduler_cpu_s",
        "schedule_makespan_s",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in _sequence(cases, "cases"):
            for mode in modes:
                structure = case["schedule_structure"][mode]
                row = {
                    field: case[field]
                    for field in (
                        "cell_id", "hardware_id", "model_id", "case_id",
                        "target_qps",
                    )
                }
                row["mode"] = mode
                for field in fields[6:]:
                    row[field] = ""
                for field in (
                    "step_count", "scheduled_tokens", "scheduled_requests",
                    "mean_scheduled_tokens_per_step",
                    "mean_scheduled_requests_per_step", "device_queue_wait_s",
                    "completion_to_update_wait_s", "predicted_device_s",
                    "launch_cpu_s", "scheduler_cpu_s", "schedule_makespan_s",
                ):
                    if field in structure:
                        row[field] = structure[field]
                if "phase_step_fraction" in structure:
                    for phase in replay.PHASES:
                        row[f"{phase}_step_fraction"] = structure[
                            "phase_step_fraction"
                        ][phase]
                writer.writerow(row)


def _render_report(payload: Mapping[str, object]) -> str:
    cases = _sequence(payload["cases"], "report cases")
    summaries = _sequence(payload["hardware_summary"], "hardware summaries")
    summary_index = {
        (str(summary["hardware_id"]), str(summary["metric"])): summary
        for summary in summaries
    }
    expected_summary_keys = {
        (hardware_id, metric)
        for hardware_id in HARDWARE_ORDER
        for metric in _METRICS
    }
    if set(summary_index) != expected_summary_keys:
        raise ValueError("report requires TTFT and TPOT summaries for three hardware")
    lines = [
        "# 三硬件 Kernel/Serving 精度与 Scheduler-depth 误差诊断",
        "",
        "本报告是 evaluation-only 诊断；production observation duration 未进入 Forward、HWSpec 或 launch selector。Serving 真值均为 dataset-native depth=2，误差分解在同一 case 的 signed metric value 上完成，MAPE 不参与物理归因。",
        "",
        "每条等式是 15 个 case 的 macro mean：每个分量先除以同一 case 的 measured metric，再跨 case 平均，因此七项与最终 prediction/measured 倍率严格闭合。`feedback` 是 Device/Launch/Scheduler 三个 trajectory feedback 的净和，不是 Scheduler CPU 时间。",
        "",
        "## 核心结论",
        "",
        "| Hardware | Metric | Macro prediction / measured | Weighted prediction / measured | Macro APE | Device direct / O | Total feedback / O | Scheduler direct / O |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for hardware in HARDWARE_ORDER:
        for metric in _METRICS:
            summary = summary_index[(hardware, metric)]
            components = summary["macro_component_over_observed"]
            hardware_cases = [
                case for case in cases if case["hardware_id"] == hardware
            ]
            macro_ape = fmean(
                abs(float(case["metrics"][metric]["final_signed_relative_error"]))
                for case in hardware_cases
            )
            feedback = sum(
                float(components[name]) for name in (
                    "device_feedback", "launch_feedback", "scheduler_feedback"
                )
            )
            lines.append(
                f"| `{hardware}` | `{metric}` | {summary['macro_final_prediction_over_observed']:.4f}× | {summary['weighted_final_prediction_over_observed']:.4f}× | {macro_ape * 100.0:.2f}% | {components['device_direct']:+.4f} | {feedback:+.4f} | {components['scheduler_direct']:+.5f} |"
            )
    a800_tpot = summary_index[("nvidia_a800", "tpot_ms")]
    a800_tpot_components = a800_tpot["macro_component_over_observed"]
    a800_tpot_error = float(a800_tpot["macro_final_signed_relative_error"])
    a800_tpot_feedback = sum(
        float(a800_tpot_components[name]) for name in (
            "device_feedback", "launch_feedback", "scheduler_feedback"
        )
    )
    c500_tpot = summary_index[("metax_c500", "tpot_ms")]
    c500_tpot_components = c500_tpot["macro_component_over_observed"]
    c500_tpot_feedback = sum(
        float(c500_tpot_components[name]) for name in (
            "device_feedback", "launch_feedback", "scheduler_feedback"
        )
    )
    ascend_tpot = summary_index[("ascend_910b", "tpot_ms")]
    ascend_tpot_components = ascend_tpot["macro_component_over_observed"]
    ascend_tpot_feedback = sum(
        float(ascend_tpot_components[name]) for name in (
            "device_feedback", "launch_feedback", "scheduler_feedback"
        )
    )
    a800_ttft = summary_index[("nvidia_a800", "ttft_ms")]
    a800_ttft_components = a800_ttft["macro_component_over_observed"]
    a800_ttft_feedback = sum(
        float(a800_ttft_components[name]) for name in (
            "device_feedback", "launch_feedback", "scheduler_feedback"
        )
    )
    c500_tpot_feedback_values = [
        sum(
            float(case["metrics"]["tpot_ms"]["component_over_observed"][name])
            for name in (
                "device_feedback", "launch_feedback", "scheduler_feedback"
            )
        )
        for case in cases
        if case["hardware_id"] == "metax_c500"
    ]
    structure_ratios = {}
    for hardware_id in HARDWARE_ORDER:
        hardware_cases = [
            case for case in cases if case["hardware_id"] == hardware_id
        ]
        structure_ratios[hardware_id] = {
            "steps": fmean(
                case["schedule_structure"]["dynamic_device_only"]["step_count"]
                / case["schedule_structure"]["observed"]["step_count"]
                for case in hardware_cases
            ),
            "requests_per_step": fmean(
                case["schedule_structure"]["dynamic_device_only"][
                    "mean_scheduled_requests_per_step"
                ]
                / case["schedule_structure"]["observed"][
                    "mean_scheduled_requests_per_step"
                ]
                for case in hardware_cases
            ),
        }
    lines.extend((
        "",
        f"A800 TPOT 的最终 `{a800_tpot['macro_final_prediction_over_observed']:.4f}×` 不是 Scheduler CPU 直接开销造成的：Device predictor 固定轨迹直接贡献 `{a800_tpot_components['device_direct']:+.4f}×O`，占最终 signed error 的 {a800_tpot_components['device_direct'] / a800_tpot_error * 100.0:.1f}%；Device duration 改变 batching/queue 后贡献 `{a800_tpot_components['device_feedback']:+.4f}×O`，三类 trajectory feedback 净贡献 `{a800_tpot_feedback:+.4f}×O`，占 {a800_tpot_feedback / a800_tpot_error * 100.0:.1f}%；Scheduler CPU 直接项只有 `{a800_tpot_components['scheduler_direct']:+.5f}×O`，约占 {a800_tpot_components['scheduler_direct'] / a800_tpot_error * 100.0:.3f}%。因此核心链路是 **Device 预测偏慢 → 动态批次/队列改变 → TPOT 再放大**。",
        "",
        f"C500 与 910B 使用相同 depth=2 机制，但原始 Device 误差方向不同：C500 TPOT 的 Device direct 为 `{c500_tpot_components['device_direct']:+.4f}×O`，feedback 净值为 `{c500_tpot_feedback:+.4f}×O`，最终为 `{c500_tpot['macro_final_prediction_over_observed']:.4f}×`；这个净值不是每个 case 的 feedback 都很小，而是 15 个 case 中 {sum(value > 0.0 for value in c500_tpot_feedback_values)} 个为正、{sum(value < 0.0 for value in c500_tpot_feedback_values)} 个为负，范围 `{min(c500_tpot_feedback_values):+.3f}` 到 `{max(c500_tpot_feedback_values):+.3f}`，存在明显抵消。910B Device direct 同样为 `{ascend_tpot_components['device_direct']:+.4f}×O`，但 feedback 达 `{ascend_tpot_feedback:+.4f}×O`，反转为最终 `{ascend_tpot['macro_final_prediction_over_observed']:.4f}×`。",
        "",
        f"A800 TTFT 的固定轨迹项存在极强的顺序依赖抵消（Device direct `{a800_ttft_components['device_direct']:+.4f}×O`，净 feedback `{a800_ttft_feedback:+.4f}×O`，最终仅 `{a800_ttft['macro_final_prediction_over_observed']:.4f}×`），这些大项不能解释为稳定的独立物理占比；它们说明固定 observed membership/order 与完整动态 admission 在低 QPS case 上差异极大。TTFT 应以最终倍率、逐 case 图和 quantile 曲线为准。",
        "",
        f"结构量与上述方向一致：Device-only 动态轨迹相对 observed trace 的平均 step 倍率为 A800 `{structure_ratios['nvidia_a800']['steps']:.3f}×`、C500 `{structure_ratios['metax_c500']['steps']:.3f}×`、910B `{structure_ratios['ascend_910b']['steps']:.3f}×`，平均 requests/step 则分别变为 `{structure_ratios['nvidia_a800']['requests_per_step']:.3f}×`、`{structure_ratios['metax_c500']['requests_per_step']:.3f}×`、`{structure_ratios['ascend_910b']['requests_per_step']:.3f}×`。但这只是跨 case 的结构摘要；个别低 QPS A800 case 会减少 step，必须结合下表逐 case 判断。",
        "",
        "## 七项闭合",
        "",
    ))
    for hardware in HARDWARE_ORDER:
        for metric in _METRICS:
            summary = summary_index[(hardware, metric)]
            components = summary["macro_component_over_observed"]
            equation = " ".join(
                f"{components[name]:+.4f} {name}" for name in _COMPONENTS
            )
            lines.extend((
                f"- `{hardware} {metric}: {summary['macro_final_prediction_over_observed']:.4f}× = 1 {equation}`",
            ))
    lines.extend((
        "",
        "## 调度结构证据",
        "",
        "下表按 `|device_feedback|` 排序，因为所列结构量对应 `O→D`（observed trace 到全动态 Device-only 反事实）；同时给出三类 trajectory feedback 的净和。Device-only 模式刻意关闭 Host launch 与 Scheduler CPU，所以其 completion→update wait 为 0，这个 0 是反事实定义，不是 full Scheduler 动态等待的预测。queue wait 是跨 step 的 critical-path 累计量，不能直接当作单请求 TTFT/TPOT。decode-only step 数与占比用于表示尾部延长/收缩。",
        "",
    ))
    hardware_ids = HARDWARE_ORDER
    for hardware in hardware_ids:
        for metric in _METRICS:
            lines.extend((
                f"### {hardware} {metric} feedback top 5",
                "",
                "| Case | Device feedback/O | Total feedback/O | Steps O→D | Tokens/step O→D | Requests/step O→D | P/M/D O→D | Device queue wait s O→D | Completion→update wait s O→D | Decode-only steps O→D |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ))
            ranked = []
            for case in cases:
                if case["hardware_id"] != hardware:
                    continue
                components = case["metrics"][metric][
                    "component_over_observed"
                ]
                feedback = sum(float(components[name]) for name in (
                    "device_feedback", "launch_feedback", "scheduler_feedback"
                ))
                device_feedback = float(components["device_feedback"])
                ranked.append((
                    abs(device_feedback), str(case["cell_id"]), case,
                    device_feedback, feedback,
                ))
            for _magnitude, _cell_id, case, device_feedback, feedback in sorted(
                ranked, reverse=True
            )[:5]:
                observed = case["schedule_structure"]["observed"]
                dynamic = case["schedule_structure"]["dynamic_device_only"]
                observed_phases = observed["phase_step_fraction"]
                dynamic_phases = dynamic["phase_step_fraction"]
                observed_counts = observed["phase_step_count"]
                dynamic_counts = dynamic["phase_step_count"]
                phase_text = (
                    f"{observed_phases['prefill'] * 100.0:.1f}/"
                    f"{observed_phases['mixed'] * 100.0:.1f}/"
                    f"{observed_phases['decode'] * 100.0:.1f}%→"
                    f"{dynamic_phases['prefill'] * 100.0:.1f}/"
                    f"{dynamic_phases['mixed'] * 100.0:.1f}/"
                    f"{dynamic_phases['decode'] * 100.0:.1f}%"
                )
                lines.append(
                    f"| `{case['model_id']}/{case['case_id']}` | {device_feedback:+.3f} | {feedback:+.3f} | {observed['step_count']}→{dynamic['step_count']} | {observed['mean_scheduled_tokens_per_step']:.1f}→{dynamic['mean_scheduled_tokens_per_step']:.1f} | {observed['mean_scheduled_requests_per_step']:.1f}→{dynamic['mean_scheduled_requests_per_step']:.1f} | {phase_text} | {observed['device_queue_wait_s']:.2f}→{dynamic['device_queue_wait_s']:.2f} | {observed['completion_to_update_wait_s']:.2f}→{dynamic['completion_to_update_wait_s']:.2f} | {observed_counts['decode']}→{dynamic_counts['decode']} |"
                )
            lines.append("")
    lines.extend((
        "## 解释边界",
        "",
        "- 七项分解遵循固定顺序 `O→T→D→F_D→L→F_L→S→F_S`；它精确闭合，但不是顺序无关的 Shapley attribution。",
        "- A800 observed Scheduler trace 每个请求包含一个额外 terminal-token work item；45-case 守恒检查在显式减去这 15×128 个 work item 后成立，client-visible TPOT denominator 已在 replay 中恢复。scheduled-request row 数不是 workload 守恒量，因为 prefill chunking 会改变同一 token 总量对应的 row 数。",
        "- 现有 v6 full-mode 产物只保留 aggregate Host timing，没有保留 full dynamic 的逐 step completion→update queue；因此表中该列只能展示 observed 与 D-only-disabled 反事实，不能据此判断 full dynamic Scheduler queue 的绝对值。",
        "- TTFT trace reconstruction 在 A800、C500、910B 的 macro contribution 分别约为 `−0.1604×O`、`−0.0719×O`、`−0.1367×O`，所以 TTFT 的 predictor 归因必须连同 reconstruction 一起解释。",
        "- Single-kernel 图第一行保留三硬件整体 log-log scatter，随后按 GEMM/MoE/FlashAttention/Vector 四个 family bucket 各给一行分桶 scatter 和一行 OP-level absolute-percentage-error CDF；CDF 将累计 OP 比例放在横轴、线性误差放在纵轴，避免整体 scatter 的对数坐标压缩相对误差。",
        "- Single-kernel 汇总对 987 个 kernel case 等权；Serving Device direct 则由真实请求的 step、shape、调用频次、通信和并行组合加权，两者方向或倍率不要求相同，不能用 kernel mean ratio 直接替代 Serving attribution。",
        "",
        "## 图表",
        "",
        "### Single-kernel dataset prediction accuracy",
        "",
        "![Three-hardware kernel latency scatter and four-family OP-level error CDFs](kernel_plots/kernel-latency-scatter.png)",
        "",
        "### Serving request/s and final mean latency prediction",
        "",
    ))
    for hardware in hardware_ids:
        lines.extend((
            f"#### {hardware}",
            "",
            f"![{hardware} Serving metrics](diagnostic_plots/{hardware}_serving_metrics.png)",
            "",
        ))
    lines.extend((
        "### Final full-serving TTFT/TPOT quantiles",
        "",
        "这些图来自完整动态 Serving 仿真器结果（`device_plus_launch_plus_scheduler`），每个子图对应一个 case，展示 measured 与 predicted 的 P0-P100 latency 曲线；它不是 fixed-trajectory replay。",
        "",
    ))
    for hardware in hardware_ids:
        lines.extend((
            f"#### {hardware}",
            "",
            f"![{hardware} final Serving TTFT quantiles](plots/{hardware}_final_serving_ttft_quantile_curves.png)",
            "",
            f"![{hardware} final Serving TPOT quantiles](plots/{hardware}_final_serving_tpot_quantile_curves.png)",
            "",
        ))
    lines.extend((
        "### Signed error breakdown",
        "",
    ))
    for hardware in hardware_ids:
        lines.extend((
            f"#### {hardware}",
            "",
            f"![{hardware} signed error breakdown](diagnostic_plots/{hardware}_error_breakdown.png)",
            "",
        ))
    lines.extend((
        "### Fixed-trajectory TTFT/TPOT quantiles",
        "",
        "这部分只用于诊断 observed scheduling trajectory 下的 Forward/replay 分布，不代表最终动态 Serving prediction。",
        "",
    ))
    for hardware in hardware_ids:
        lines.extend((
            f"#### {hardware}",
            "",
            f"![{hardware} TTFT quantiles](plots/{hardware}_scheduling_trajectory_replay_ttft_quantile_curves.png)",
            "",
            f"![{hardware} TPOT quantiles](plots/{hardware}_scheduling_trajectory_replay_tpot_quantile_curves.png)",
            "",
        ))
    lines.extend((
        "## 可复核产物",
        "",
        "- `breakdown.json`: 45 case、90 metric rows 的 O/T/D/L/S/F_D/F_L/F_S 与七项分解。",
        "- `per_case_breakdown.csv`: 逐 case signed delta、delta/O、signed error share 与 closure residual。",
        "- `hardware_summary.csv`: hardware×metric 的 macro/median/P90/weighted 汇总。",
        "- `schedule_structure.csv`: observed 与三个动态模式的 step/batch/phase/queue 结构。",
        "- `hashes.json`: 所有报告产物的 SHA256。",
        "",
    ))
    return "\n".join(lines)


def _observed_mean(observed: Mapping[str, object], metric: str) -> float:
    latency = _mapping(observed["latencies"], "observed latency")
    return _finite(latency[metric]["mean"], f"observed {metric}")


def _replay_mean(metrics: Mapping[str, object], metric: str) -> float:
    return _finite(metrics["latency"][metric]["mean"], f"replay {metric}")


def _percentile(sorted_values: Sequence[float], percentile: int) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()






def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return value




def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{label} must be a non-negative integer")
    return value
