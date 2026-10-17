"""Deterministic Markdown reporting for production-Trace sweeps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re

from evaluation.serving.metrics.interactivity_tgs import (
    arrival_rate_sweep_analysis_manifest,
    InteractivityTGSContract,
    sweep_metric_point_from_manifest,
)
from evaluation.serving.metrics.slo_goodput import SLOContract
from evaluation.serving.metrics.scheduler_progress import (
    validate_scheduler_progress_manifest,
)
from evaluation.serving.simulation.execution_artifact import (
    performance_timing_status,
    PROJECTED_PERFORMANCE_TIMING_STATUS,
    UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
)
from src.serving.backend.vllm import normalize_mtp_scenario_manifest
from src.serving.host_timing import (
    LAUNCH_KINDS,
    STEP_PHASES,
    validate_serving_timing_mode,
)

from evaluation.serving.metrics.sharegpt_compact_evaluation import (
    EXECUTION_ARTIFACT_FIELDS,
)

from evaluation.json_shape import require_object as _mapping


_SWEEP_FIELDS = {
    "schema_version",
    "kind",
    "interpretation",
    "interactivity_tgs",
    "peak_selection",
    "saturation_analysis",
    "scale_order",
    "arrival_transform",
    "point_count",
    "arrival_rate_scales",
    "source_trace",
    "execution_provenance",
    "points",
}
_AGGREGATE_SWEEP_FIELDS = _SWEEP_FIELDS | {"persistence"}
_AGGREGATE_RESULT_FIELDS = {
    "schema_version",
    "kind",
    "persistence",
    "version",
    "backend",
    "deployment_id",
    "hardware_ids",
    "model_compute_dtype",
    "model_config_provenance",
    "deployment",
    "deployment_plan",
    "routing_policy",
    "kv_handoff_mode",
    "kv_capacity",
    "kv_lifecycle_replay",
    "launch_policy",
    "prediction_method",
    "execution_mode",
    "execution_cache_stats",
    "execution_parallelism",
    "serving_timing_mode",
    "host_timing",
    "performance_timing_status",
    "execution_provenance",
    "workload_assumption_binding",
    "main_chain_status",
    "execution_artifact",
    "scheduler_progress",
    "scheduler_runtime",
    "mtp_scenario",
    "mtp_acceptance_outcomes",
    "mtp_pd_handoff_contract",
    "arrival_process",
    "num_steps",
    "num_transfers",
    "num_kernel_calls",
    "schedule_end_time_s",
    "simulation_wall_time_s",
    "evaluation_wall_time_breakdown_s",
    "process_peak_rss_bytes",
    "metrics",
    "request_source",
}
_POINT_FIELDS = {
    "arrival_rate_scale",
    "request_stream_sha256",
    "validation_status",
    "metrics",
    "result",
}
_SUMMARY_METRIC_FIELDS = {
    "request_throughput_per_s",
    "output_token_throughput_per_s",
    "token_throughput_per_s",
    "latency_distributions",
    "slo_goodput",
}
_SOURCE_TRACE_FIELDS = {
    "schema_version",
    "successful_only",
    "source_path",
    "source_sha256",
    "source_format",
    "field_mapping",
    "input_time_unit",
    "normalized_time_unit",
    "request_id_policy",
    "equal_arrival_order",
    "invalid_row_policy",
    "unmapped_field_policy",
    "prefix_oracle",
    "scheduler_prefix_alignment_owner",
    "success_filter",
    "statistics",
    "normalized_request_stream_sha256",
}
_REPLAY_FIELDS = {
    "schema_version",
    "kind",
    "arrival_rate_scale",
    "arrival_transform",
    "arrival_anchor_s",
    "request_count",
    "arrival_time_total_s",
    "arrival_time_min_s",
    "arrival_time_max_s",
    "request_distribution_policy",
    "interpretation",
    "request_stream_sha256",
}
_EXECUTION_PROVENANCE_FIELDS = {
    "schema_version",
    "repository",
    "hardware_configs",
    "loaded_hardware_ids",
    "performance_timing_status",
    "runtime_inputs",
    "launch_inputs",
    "microbenchmark_inputs",
    "prediction_method",
    "external_predictor_inputs",
}
_EXPECTED_FIXED_STATUS = {
    "trace_replay": "validated_conservation",
    "scheduler_control_plane": "native_pinned_vllm",
}
_ARRIVAL_TRANSFORM = "t_anchor + (t - t_anchor) / arrival_rate_scale"


def render_production_trace_sweep_report(
    payload: Mapping[str, object],
) -> str:
    """Validate a production-Trace sweep v2 and render it as Markdown.

    This is a read-only report consumer.  It does not recompute predictions or
    metric values, and it preserves the validated timing-support status.
    """

    sweep = _mapping(payload, "production Trace sweep")
    if ("schema_version" in sweep
            and sweep["schema_version"] == 3):
        return _render_aggregate_production_trace_sweep_report(sweep)
    _exact_fields(sweep, _SWEEP_FIELDS, "production Trace sweep")
    if sweep["schema_version"] != 2:
        raise ValueError("production Trace sweep schema_version must be 2")
    if sweep["kind"] != "production_trace_arrival_rate_sweep":
        raise ValueError(
            "production Trace report requires kind "
            "'production_trace_arrival_rate_sweep'")
    if sweep["interpretation"] != "successful_path_performance_only":
        raise ValueError(
            "production Trace sweep interpretation must be "
            "'successful_path_performance_only'")
    if sweep["scale_order"] != "caller_supplied":
        raise ValueError("production Trace sweep scale_order must be caller_supplied")
    if sweep["arrival_transform"] != _ARRIVAL_TRANSFORM:
        raise ValueError("production Trace sweep arrival transform is unsupported")

    source_trace = _validate_source_trace(sweep["source_trace"])
    execution_provenance = _validate_execution_provenance(
        sweep["execution_provenance"])
    points = _sequence(sweep["points"], "production Trace sweep points")
    if not points:
        raise ValueError("production Trace sweep must contain at least one point")
    point_count = _nonnegative_integer(
        sweep["point_count"], "production Trace sweep point_count")
    if point_count != len(points):
        raise ValueError("production Trace sweep point_count disagrees with points")
    scales = tuple(
        _positive_number(scale, "production Trace arrival-rate scale")
        for scale in _sequence(
            sweep["arrival_rate_scales"],
            "production Trace arrival-rate scales",
        )
    )
    if len(scales) != len(points):
        raise ValueError(
            "production Trace arrival-rate scales disagree with point_count")
    if len(scales) != len(set(scales)):
        raise ValueError("production Trace arrival-rate scales must be unique")

    validated_points = tuple(
        _validate_point(point, index, scales[index], source_trace,
                        execution_provenance)
        for index, point in enumerate(points)
    )
    timing_status = execution_provenance["performance_timing_status"]
    if any(
            point["result"]["performance_timing_status"] != timing_status
            for point in validated_points):
        raise ValueError(
            "production Trace sweep points disagree on performance timing "
            "status")
    mtp_scenario = validated_points[0]["result"]["mtp_scenario"]
    if any(
            point["result"]["mtp_scenario"] != mtp_scenario
            for point in validated_points[1:]):
        raise ValueError(
            "production Trace sweep points disagree on the MTP scenario")
    _validate_sweep_analysis(sweep, validated_points)

    lines = [
        "# Production Trace Serving Sweep Report",
        "",
        "## Scope and validation status",
        "",
        *_render_timing_scope(timing_status),
        "- Scheduler control plane: `native_pinned_vllm`.",
        "- Trace replay: `validated_conservation`.",
        *_render_mtp_scenario(mtp_scenario),
        "- Interpretation: successful-path performance only; failed source "
        "requests are not replayed.",
        "- This report does not claim device-runtime performance validation.",
        "",
        "## Reproducibility identity",
        "",
        f"- Source path: {_code(source_trace['source_path'])}",
        f"- Source SHA256: `{source_trace['source_sha256']}`",
        "- Normalized source request-stream SHA256: "
        f"`{source_trace['normalized_request_stream_sha256']}`",
    ]
    lines.extend(_render_execution_provenance(execution_provenance))
    lines.extend(_render_sweep_analysis(sweep))
    for index, point in enumerate(validated_points, start=1):
        lines.extend(_render_point(index, point))
    return "\n".join(lines) + "\n"


def write_production_trace_sweep_plots(
    output_dir: str | Path,
    payload: Mapping[str, object],
) -> Mapping[str, Path]:
    """Write maintained throughput, latency, and simulation-speed plots."""

    render_production_trace_sweep_report(payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    sweep = _mapping(payload, "production Trace sweep")
    points = tuple(
        _mapping(point, f"production Trace plot point {index}")
        for index, point in enumerate(
            _sequence(sweep["points"], "production Trace plot points"))
    )
    scales = tuple(
        _positive_number(
            point["arrival_rate_scale"],
            "production Trace plot arrival-rate scale",
        )
        for point in points
    )
    metrics = tuple(
        _mapping(point["metrics"], "production Trace plot metrics")
        for point in points
    )
    results = tuple(
        _mapping(point["result"], "production Trace plot result")
        for point in points
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Production Trace plot generation requires matplotlib") from error

    paths = {
        "throughput": destination / "production_trace_throughput.png",
        "ttft_p0_p100": destination / "production_trace_ttft_p0_p100.png",
        "tpot_p0_p100": destination / "production_trace_tpot_p0_p100.png",
        "simulation_speed": destination / "production_trace_simulation_speed.png",
        "interactivity_tgs": destination / "production_trace_interactivity_tgs.png",
    }
    positions = tuple(range(len(points)))
    labels = tuple(f"{scale:g}×" for scale in scales)

    figure, axis = plt.subplots(figsize=(max(8.0, len(points) * 1.2), 5.5))
    width = 0.36
    request_values = tuple(
        _nonnegative_number(
            row["request_throughput_per_s"], "request throughput")
        for row in metrics
    )
    token_values = tuple(
        _nonnegative_number(
            row["output_token_throughput_per_s"], "output-token throughput")
        for row in metrics
    )
    axis.bar(
        [position - width / 2 for position in positions],
        request_values,
        width=width,
        label="Requests/s",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        token_values,
        width=width,
        label="Output tokens/s",
    )
    axis.set_title("Production Trace throughput by arrival-rate scale")
    axis.set_xlabel("Arrival-rate scale")
    axis.set_ylabel("Throughput per second")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["throughput"], dpi=160)
    plt.close(figure)

    for metric_name, path_key, title in (
        ("ttft_s", "ttft_p0_p100", "TTFT P0–P100"),
        ("tpot_s", "tpot_p0_p100", "TPOT P0–P100"),
    ):
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        for scale, row in zip(scales, metrics, strict=True):
            distributions = _mapping(
                row["latency_distributions"], "latency distributions")
            distribution = _mapping(
                distributions[metric_name], f"{metric_name} distribution")
            if distribution["availability"] != "available":
                continue
            quantiles = _mapping(
                distribution["quantiles"], f"{metric_name} quantiles")
            axis.plot(
                _sequence(quantiles["percentiles"], "latency percentiles"),
                _sequence(quantiles["values_s"], "latency quantile values"),
                label=f"{scale:g}× arrival rate",
                linewidth=1.8,
            )
        axis.set_title(title)
        axis.set_xlabel("Percentile")
        axis.set_ylabel("Seconds")
        axis.set_xlim(0, 100)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(paths[path_key], dpi=160)
        plt.close(figure)

    simulation_wall = tuple(
        _nonnegative_number(
            result["simulation_wall_time_s"], "simulation wall time")
        for result in results
    )
    virtual_time = tuple(
        _nonnegative_number(
            result["schedule_end_time_s"], "virtual schedule end time")
        for result in results
    )
    figure, axis = plt.subplots(figsize=(max(8.0, len(points) * 1.2), 5.5))
    bars = axis.bar(positions, simulation_wall, color="#4C78A8")
    axis.bar_label(
        bars,
        labels=[f"{value:.2f}s" for value in simulation_wall],
        padding=3,
    )
    for position, wall, virtual in zip(
        positions, simulation_wall, virtual_time, strict=True
    ):
        ratio = math.inf if virtual == 0.0 else wall / virtual
        label = "n/a" if not math.isfinite(ratio) else f"{ratio:.2f} wall/virtual"
        axis.annotate(
            label,
            (position, wall),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axis.set_title("Production Trace simulation wall time")
    axis.set_xlabel("Arrival-rate scale")
    axis.set_ylabel("Wall seconds")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths["simulation_speed"], dpi=160)
    plt.close(figure)

    # The trade-off contract is persisted in the sweep JSON, so this plot
    # consumes only the validated P99 TPOT/interactivity and TGS values.
    tradeoff = _mapping(sweep["interactivity_tgs"], "Interactivity-TGS analysis")
    if tradeoff["availability"] != "unavailable_no_metric_contract":
        points_tradeoff = tuple(
            _mapping(point, "Interactivity-TGS plot point")
            for point in _sequence(tradeoff["points"], "Interactivity-TGS plot points")
        )
        frontier_tradeoff = tuple(
            _mapping(point, "Interactivity-TGS frontier point")
            for point in _sequence(tradeoff["pareto_frontier"], "Interactivity-TGS frontier")
        )
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        x_values = tuple(_nonnegative_number(point["interactivity_value_s"], "interactivity") for point in points_tradeoff)
        y_values = tuple(_nonnegative_number(point["tgs_value"], "TGS") for point in points_tradeoff)
        labels_tradeoff = tuple(_positive_number(point["arrival_rate_scale"], "arrival-rate scale") for point in points_tradeoff)
        axis.scatter(x_values, y_values, s=55, label="Sweep points")
        for x_value, y_value, label in zip(x_values, y_values, labels_tradeoff, strict=True):
            axis.annotate(f"{label:g}×", (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
        if frontier_tradeoff:
            frontier_xy = tuple(sorted(
                (_nonnegative_number(point["interactivity_value_s"], "frontier interactivity"), _nonnegative_number(point["tgs_value"], "frontier TGS"))
                for point in frontier_tradeoff
            ))
            axis.plot(tuple(item[0] for item in frontier_xy), tuple(item[1] for item in frontier_xy), color="#E45756", marker="o", linewidth=2, label="Pareto frontier")
        axis.set_title("Interactivity (P99 TPOT) vs TGS")
        axis.set_xlabel("P99 TPOT (seconds)")
        axis.set_ylabel("Output-token TGS (tokens/s)")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(paths["interactivity_tgs"], dpi=160)
        plt.close(figure)
    return paths


def _render_aggregate_production_trace_sweep_report(
    sweep: Mapping[str, object],
) -> str:
    """Validate and render the request-safe persisted sweep schema."""

    _exact_fields(
        sweep, _AGGREGATE_SWEEP_FIELDS, "aggregate production Trace sweep")
    if sweep["kind"] != "production_trace_arrival_rate_sweep":
        raise ValueError("aggregate production Trace sweep kind is invalid")
    persistence = _mapping(
        sweep["persistence"], "aggregate persistence contract")
    _exact_fields(
        persistence,
        {"schema_version", "projection", "request_level_content"},
        "aggregate persistence contract",
    )
    if persistence != {
            "schema_version": 1,
            "projection": "aggregate_only_production_trace_v1",
            "request_level_content": "omitted",
    }:
        raise ValueError("aggregate persistence contract is not canonical")
    if sweep["interpretation"] != "successful_path_performance_only":
        raise ValueError("aggregate production Trace interpretation is invalid")
    if sweep["scale_order"] != "caller_supplied":
        raise ValueError("aggregate production Trace scale order is invalid")
    if sweep["arrival_transform"] != _ARRIVAL_TRANSFORM:
        raise ValueError("aggregate production Trace arrival transform is invalid")
    source_trace = _validate_source_trace(sweep["source_trace"])
    execution_provenance = _validate_execution_provenance(
        sweep["execution_provenance"])
    points = _sequence(sweep["points"], "aggregate production Trace points")
    scales = tuple(
        _positive_number(scale, "aggregate production Trace scale")
        for scale in _sequence(
            sweep["arrival_rate_scales"],
            "aggregate production Trace scales",
        )
    )
    if (not points
            or len(points) != len(scales)
            or sweep["point_count"] != len(points)
            or len(scales) != len(set(scales))):
        raise ValueError("aggregate production Trace point cardinality is invalid")
    validated_points = tuple(
        _validate_aggregate_point(
            point,
            index,
            scales[index],
            source_trace,
            execution_provenance,
        )
        for index, point in enumerate(points)
    )
    _validate_sweep_analysis(sweep, validated_points)
    timing_status = execution_provenance["performance_timing_status"]
    mtp_scenario = validated_points[0]["result"]["mtp_scenario"]
    if any(
            point["result"]["mtp_scenario"] != mtp_scenario
            for point in validated_points[1:]):
        raise ValueError(
            "aggregate production Trace points disagree on MTP scenario")
    lines = [
        "# Production Trace Serving Sweep Report",
        "",
        "## Scope and validation status",
        "",
        *_render_timing_scope(timing_status),
        "- Persistence: **aggregate-only**; request-level arrivals, metrics, "
        "routes, transfers, Scheduler steps, and MTP outcomes are omitted.",
        "- Scheduler control plane: `native_pinned_vllm`.",
        "- Trace replay: `validated_conservation`.",
        *_render_mtp_scenario(mtp_scenario),
        "- Interpretation: successful-path performance only.",
        "",
        "## Reproducibility identity",
        "",
        f"- Source SHA256: `{source_trace['source_sha256']}`",
        "- Normalized source request-stream SHA256: "
        f"`{source_trace['normalized_request_stream_sha256']}`",
    ]
    lines.extend(_render_execution_provenance(execution_provenance))
    lines.extend(_render_sweep_analysis(sweep))
    for index, point in enumerate(validated_points, start=1):
        lines.extend(_render_aggregate_point(index, point))
    return "\n".join(lines) + "\n"


def _validate_aggregate_point(
    value: object,
    index: int,
    expected_scale: float,
    source_trace: Mapping[str, object],
    execution_provenance: Mapping[str, object],
) -> Mapping[str, object]:
    label = f"aggregate production Trace point {index}"
    point = _mapping(value, label)
    _exact_fields(point, _POINT_FIELDS, label)
    scale = _positive_number(point["arrival_rate_scale"], f"{label} scale")
    if scale != expected_scale:
        raise ValueError(f"{label} scale disagrees with sweep scales")
    stream_sha256 = _sha256(
        point["request_stream_sha256"], f"{label} request stream SHA256")
    status = _mapping(point["validation_status"], f"{label} status")
    _exact_fields(
        status,
        set(_EXPECTED_FIXED_STATUS) | {"performance_timing"},
        f"{label} status",
    )
    if any(status[name] != expected for name, expected in (
            _EXPECTED_FIXED_STATUS.items())):
        raise ValueError(f"{label} fixed validation status is invalid")
    summary_metrics = _mapping(point["metrics"], f"{label} summary metrics")
    _exact_fields(
        summary_metrics, _SUMMARY_METRIC_FIELDS, f"{label} summary metrics")
    result = _mapping(point["result"], f"{label} result")
    _exact_fields(result, _AGGREGATE_RESULT_FIELDS, f"{label} result")
    if (result["schema_version"] != 1
            or result["kind"] != "production_trace_simulation_aggregate"
            or result["version"] != 3
            or result["backend"] != "vllm"):
        raise ValueError(f"{label} aggregate result identity is invalid")
    result_persistence = _mapping(
        result["persistence"], f"{label} result persistence")
    _exact_fields(
        result_persistence,
        {"projection", "request_level_content", "full_in_memory_result_sha256"},
        f"{label} result persistence",
    )
    if (result_persistence["projection"]
            != "aggregate_only_production_trace_v1"
            or result_persistence["request_level_content"] != "omitted"):
        raise ValueError(f"{label} aggregate projection is invalid")
    _sha256(
        result_persistence["full_in_memory_result_sha256"],
        f"{label} in-memory result SHA256",
    )
    expected_timing = performance_timing_status(
        result["launch_policy"],
        tuple(_sequence(result["hardware_ids"], f"{label} hardware IDs")),
    )
    if (result["performance_timing_status"] != expected_timing
            or status["performance_timing"] != expected_timing
            or execution_provenance["performance_timing_status"]
            != expected_timing
            or result["execution_provenance"] != execution_provenance):
        raise ValueError(f"{label} timing or execution provenance is invalid")
    _validate_execution_and_host_timing(result, label)
    request_source = _mapping(
        result["request_source"], f"{label} request source")
    _exact_fields(
        request_source,
        {"schema_version", "kind", "trace", "replay",
         "zero_output_contract", "prefix_alignment"},
        f"{label} request source",
    )
    if (request_source["kind"] != "production_trace"
            or request_source["trace"] != source_trace):
        raise ValueError(f"{label} source Trace identity is invalid")
    replay = _mapping(request_source["replay"], f"{label} replay")
    _exact_fields(replay, _REPLAY_FIELDS, f"{label} replay")
    if (replay["request_stream_sha256"] != stream_sha256
            or replay["arrival_rate_scale"] != scale
            or replay["kind"] != "arrival_rate_scale"):
        raise ValueError(f"{label} replay identity is invalid")
    alignment = _mapping(
        request_source["prefix_alignment"], f"{label} prefix alignment")
    _exact_fields(
        alignment,
        {"schema_version", "owner", "vllm_version", "rule",
         "kv_block_size_tokens", "aggregate",
         "request_level_entries_sha256", "request_level_entries"},
        f"{label} prefix alignment",
    )
    if alignment["request_level_entries"] != "omitted":
        raise ValueError(f"{label} retained request-level prefix entries")
    _sha256(
        alignment["request_level_entries_sha256"],
        f"{label} prefix-entry SHA256",
    )
    aggregate_metrics = _mapping(result["metrics"], f"{label} metrics")
    _exact_fields(
        aggregate_metrics,
        _SUMMARY_METRIC_FIELDS | {
            "system_resources_sha256", "critical_path_sha256",
            "request_level_metrics",
        },
        f"{label} metrics",
    )
    if aggregate_metrics["request_level_metrics"] != "omitted":
        raise ValueError(f"{label} retained request-level metrics")
    for name in _SUMMARY_METRIC_FIELDS:
        if aggregate_metrics[name] != summary_metrics[name]:
            raise ValueError(f"{label} summary metric {name} disagrees")
    for name in (
            "request_throughput_per_s",
            "output_token_throughput_per_s",
            "token_throughput_per_s"):
        _nonnegative_number(aggregate_metrics[name], f"{label} {name}")
    _nonnegative_number(
        result["simulation_wall_time_s"], f"{label} simulation wall time")
    _validate_latency_distributions(
        aggregate_metrics["latency_distributions"], label)
    _validate_slo_goodput(aggregate_metrics["slo_goodput"], label)
    _sha256(aggregate_metrics["system_resources_sha256"],
            f"{label} system-resource SHA256")
    _sha256(aggregate_metrics["critical_path_sha256"],
            f"{label} critical-path SHA256")
    artifact = _mapping(result["execution_artifact"], f"{label} artifact")
    _exact_fields(
        artifact,
        {"schema_version", "hash_method", "performance_timing_status",
         "counts", "artifact_sha256", "request_level_artifact"},
        f"{label} artifact",
    )
    if artifact["request_level_artifact"] != "omitted":
        raise ValueError(f"{label} retained request-level execution artifact")
    artifact_sha256 = _sha256(
        artifact["artifact_sha256"], f"{label} artifact SHA256")
    scheduler_progress = validate_scheduler_progress_manifest(
        result["scheduler_progress"])
    if scheduler_progress["summary"]["step_count"] != result["num_steps"]:
        raise ValueError(f"{label} Scheduler progress step count disagrees")
    main_chain = _mapping(result["main_chain_status"], f"{label} main chain")
    _exact_fields(
        main_chain, {"status", "chain", "execution_artifact_sha256"},
        f"{label} main chain",
    )
    if (main_chain["status"] != "executed"
            or main_chain["execution_artifact_sha256"] != artifact_sha256):
        raise ValueError(f"{label} main-chain attestation is invalid")
    binding = _mapping(
        result["workload_assumption_binding"], f"{label} binding")
    _exact_fields(
        binding,
        {"schema_version", "request_stream_sha256",
         "expert_routing_policy_sha256", "mtp_acceptance_policy_sha256",
         "composite_sha256", "independence_contract"},
        f"{label} binding",
    )
    if binding["request_stream_sha256"] != stream_sha256:
        raise ValueError(f"{label} workload binding request hash disagrees")
    components = {
        "request_stream_sha256": binding["request_stream_sha256"],
        "expert_routing_policy_sha256": (
            binding["expert_routing_policy_sha256"]),
        "mtp_acceptance_policy_sha256": (
            binding["mtp_acceptance_policy_sha256"]),
    }
    expected_composite = hashlib.sha256(
        _canonical_json_bytes(components)).hexdigest()
    if binding["composite_sha256"] != expected_composite:
        raise ValueError(f"{label} composite workload binding is invalid")
    try:
        scenario = normalize_mtp_scenario_manifest(result["mtp_scenario"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} MTP scenario is invalid: {error}") from error
    if _canonical_json_bytes(scenario) != _canonical_json_bytes(
            result["mtp_scenario"]):
        raise ValueError(f"{label} MTP scenario is not canonical")
    return point


def _render_aggregate_point(
    index: int,
    point: Mapping[str, object],
) -> list[str]:
    result = _mapping(point["result"], "aggregate point result")
    metrics = _mapping(result["metrics"], "aggregate point metrics")
    artifact = _mapping(result["execution_artifact"], "aggregate artifact")
    lines = [
        f"## Sweep point {index}: arrival-rate scale "
        f"{_format(point['arrival_rate_scale'])}",
        "",
        f"- Request-stream SHA256: `{point['request_stream_sha256']}`",
        f"- Execution artifact SHA256: `{artifact['artifact_sha256']}`",
        f"- Scheduler steps: `{result['num_steps']}`; kernel calls: "
        f"`{result['num_kernel_calls']}`; transfers: "
        f"`{result['num_transfers']}`.",
        *_render_timing_scope(result["performance_timing_status"]),
        "",
        "### Throughput",
        "",
        "| Metric | Value per second |",
        "|---|---:|",
        f"| Requests | {_format(metrics['request_throughput_per_s'])} |",
        "| Output tokens | "
        f"{_format(metrics['output_token_throughput_per_s'])} |",
        "| Input + output tokens | "
        f"{_format(metrics['token_throughput_per_s'])} |",
        "",
    ]
    lines.extend(_render_latency_distributions(metrics["latency_distributions"]))
    lines.extend(_render_slo(metrics["slo_goodput"]))
    return lines


def _validate_execution_and_host_timing(
    result: Mapping[str, object],
    label: str,
) -> None:
    execution_mode = result["execution_mode"]
    if execution_mode not in ("accelerated", "reference"):
        raise ValueError(f"{label} execution mode is invalid")
    mode = validate_serving_timing_mode(result["serving_timing_mode"])
    hardware_ids = tuple(
        _string(item, f"{label} hardware ID")
        for item in _sequence(result["hardware_ids"], f"{label} hardware IDs")
    )
    if not hardware_ids or len(hardware_ids) != len(set(hardware_ids)):
        raise ValueError(f"{label} hardware IDs must be non-empty and unique")
    timing = _mapping(result["host_timing"], f"{label} Host timing")
    _exact_fields(timing, {
        "schema_version", "timing_mode", "scheduler_measurement_boundary",
        "launch_measurement_boundary", "profiles", "runtime_prepare_profiles",
        "aggregate", "steps",
    }, f"{label} Host timing")
    if timing["schema_version"] != 1 or timing["timing_mode"] != mode:
        raise ValueError(f"{label} Host timing schema or mode is invalid")
    if timing["scheduler_measurement_boundary"] != (
            "in_scheduler_process_schedule_and_output_update_only"):
        raise ValueError(f"{label} Host timing Scheduler boundary is invalid")
    if timing["launch_measurement_boundary"] != (
            "host_submit_only_no_device_sync"):
        raise ValueError(f"{label} Host timing launch boundary is invalid")

    profiles = tuple(
        _validate_host_launch_profile_manifest(profile, label)
        for profile in _sequence(
            timing["profiles"], f"{label} Host launch profiles")
    )
    profile_hardware_ids = tuple(profile["hardware_id"] for profile in profiles)
    if mode == "device_only":
        expected_profile_hardware_ids = ()
    elif mode in ("device_plus_launch", "device_plus_launch_plus_scheduler"):
        expected_profile_hardware_ids = tuple(sorted(hardware_ids))
    else:
        raise AssertionError("validated Serving timing mode is unknown")
    if profile_hardware_ids != expected_profile_hardware_ids:
        raise ValueError(
            f"{label} Host launch profiles disagree with timing mode or hardware")
    runtime_prepare_profiles = tuple(
        _validate_host_runtime_prepare_profile_manifest(profile, label)
        for profile in _sequence(
            timing["runtime_prepare_profiles"],
            f"{label} Host runtime prepare profiles")
    )
    runtime_prepare_hardware_ids = tuple(
        profile["hardware_id"] for profile in runtime_prepare_profiles)
    if any(hardware_id not in hardware_ids
           for hardware_id in runtime_prepare_hardware_ids):
        raise ValueError(
            f"{label} Host runtime prepare profile hardware is not deployed")

    aggregate = _mapping(
        timing["aggregate"], f"{label} Host timing aggregate")
    summed_fields = {
        "predicted_device_s", "launch_cpu_s", "launch_critical_path_s",
        "host_device_overlap_s", "scheduler_plan_cpu_s",
        "scheduler_update_cpu_s", "execution_wall_s", "step_wall_s",
        "eager_dispatch_count", "graph_replay_count",
    }
    aggregate_fields = summed_fields | {
        "scheduler_cpu_s", "schedule_makespan_s", "step_count",
    }
    _exact_fields(aggregate, aggregate_fields, f"{label} Host timing aggregate")
    count_fields = {
        "eager_dispatch_count", "graph_replay_count", "step_count",
    }
    for field in aggregate_fields - count_fields:
        _nonnegative_number(aggregate[field], f"{label} Host timing {field}")
    for field in count_fields:
        _nonnegative_integer(aggregate[field], f"{label} Host timing {field}")
    if not math.isclose(
            aggregate["scheduler_cpu_s"],
            aggregate["scheduler_plan_cpu_s"]
            + aggregate["scheduler_update_cpu_s"],
            rel_tol=1e-12,
            abs_tol=1e-15):
        raise ValueError(f"{label} Host Scheduler CPU aggregate is inconsistent")

    steps = tuple(
        _validate_host_timing_step(step, label)
        for step in _sequence(timing["steps"], f"{label} Host timing steps")
    )
    if not steps or len(steps) != aggregate["step_count"]:
        raise ValueError(f"{label} Host timing step count is inconsistent")
    for field in summed_fields:
        observed_sum = sum(step[field] for step in steps)
        if not math.isclose(
                aggregate[field], observed_sum, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"{label} Host timing aggregate {field} is inconsistent")


def _validate_host_launch_profile_manifest(
    value: object,
    label: str,
) -> Mapping[str, object]:
    profile = _mapping(value, f"{label} Host launch profile")
    if "schema_version" not in profile:
        raise ValueError(f"{label} Host launch profile lacks schema_version")
    schema_version = profile["schema_version"]
    if type(schema_version) is not int:
        raise TypeError(f"{label} Host launch profile schema must be an integer")
    common_fields = {
        "schema_version", "hardware_id", "host_cpu", "operating_system",
        "driver_version", "runtime", "framework", "service_time_s",
        "measurement", "source",
    }
    if schema_version == 1:
        _exact_fields(
            profile,
            common_fields | {"phase_launch_kind"},
            f"{label} Host launch profile",
        )
        phases = _mapping(
            profile["phase_launch_kind"],
            f"{label} Host launch phase mapping",
        )
        _exact_fields(
            phases, set(STEP_PHASES),
            f"{label} Host launch phase mapping",
        )
        if any(phases[phase] not in LAUNCH_KINDS for phase in STEP_PHASES):
            raise ValueError(f"{label} Host launch phase mapping is invalid")
    elif schema_version == 2:
        # ``device_launch_gap`` is the schema's one optional section.
        _exact_fields(
            profile,
            common_fields | (
                {"device_launch_gap"} if "device_launch_gap" in profile
                else set()),
            f"{label} Host launch profile",
        )
    else:
        raise ValueError(f"{label} Host launch profile schema is invalid")
    for field in (
            "hardware_id", "host_cpu", "operating_system", "driver_version"):
        _string(profile[field], f"{label} Host launch profile {field}")
    for field in ("runtime", "framework"):
        identity = _mapping(
            profile[field], f"{label} Host launch profile {field}")
        _exact_fields(
            identity, {"name", "version"},
            f"{label} Host launch profile {field}")
        _string(identity["name"], f"{label} Host launch profile {field} name")
        _string(
            identity["version"], f"{label} Host launch profile {field} version")
    service = _mapping(
        profile["service_time_s"], f"{label} Host launch service time")
    _exact_fields(
        service, {"eager_enqueue", "graph_replay_enqueue"},
        f"{label} Host launch service time")
    for field in ("eager_enqueue", "graph_replay_enqueue"):
        _positive_number(service[field], f"{label} Host launch {field}")
    measurement = _mapping(
        profile["measurement"], f"{label} Host launch measurement")
    _exact_fields(
        measurement, {"artifact", "sha256", "boundary"},
        f"{label} Host launch measurement")
    _string(measurement["artifact"], f"{label} Host launch artifact")
    _sha256(measurement["sha256"], f"{label} Host launch measurement SHA256")
    if measurement["boundary"] != "host_submit_only_no_device_sync":
        raise ValueError(f"{label} Host launch measurement boundary is invalid")
    source = _mapping(profile["source"], f"{label} Host launch source")
    _exact_fields(
        source, {"commit", "path", "sha256"}, f"{label} Host launch source")
    commit = _string(source["commit"], f"{label} Host launch source commit")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"{label} Host launch source commit is invalid")
    _string(source["path"], f"{label} Host launch source path")
    _sha256(source["sha256"], f"{label} Host launch source SHA256")
    return profile


def _validate_host_runtime_prepare_profile_manifest(
    value: object,
    label: str,
) -> Mapping[str, object]:
    profile = _mapping(value, f"{label} Host runtime prepare profile")
    _exact_fields(profile, {
        "hardware_id", "service_time_s_by_mechanism",
        "lane_service_time_s_by_mechanism", "profile_path", "profile_sha256",
    }, f"{label} Host runtime prepare profile")
    _string(profile["hardware_id"], f"{label} runtime prepare hardware ID")
    services = _mapping(
        profile["service_time_s_by_mechanism"],
        f"{label} runtime prepare services")
    if not services:
        raise ValueError(f"{label} runtime prepare services are empty")
    for mechanism, service_time_s in services.items():
        _string(mechanism, f"{label} runtime prepare mechanism")
        _positive_number(
            service_time_s, f"{label} runtime prepare mechanism service")
    lanes = _sequence(
        profile["lane_service_time_s_by_mechanism"],
        f"{label} runtime prepare lane services")
    if lanes:
        raise ValueError(
            f"{label} runtime prepare lane services are unsupported in this report")
    _string(profile["profile_path"], f"{label} runtime prepare profile path")
    _sha256(profile["profile_sha256"], f"{label} runtime prepare profile SHA256")
    return profile


def _validate_host_timing_step(
    value: object,
    label: str,
) -> Mapping[str, object]:
    step = _mapping(value, f"{label} Host timing step")
    count_fields = {
        "scheduled_request_count", "prefill_token_count", "decode_token_count",
        "eager_dispatch_count", "graph_replay_count",
    }
    duration_fields = {
        "predicted_device_s", "launch_cpu_s", "launch_critical_path_s",
        "host_device_overlap_s", "scheduler_plan_cpu_s",
        "scheduler_update_cpu_s", "execution_wall_s", "step_wall_s",
    }
    _exact_fields(step, {
        "step_id", "pool_name", "phase", "launch_kinds",
    } | count_fields | duration_fields, f"{label} Host timing step")
    _string(step["step_id"], f"{label} Host timing step ID")
    _string(step["pool_name"], f"{label} Host timing pool name")
    if step["phase"] not in STEP_PHASES:
        raise ValueError(f"{label} Host timing phase is invalid")
    for field in count_fields:
        _nonnegative_integer(step[field], f"{label} Host timing step {field}")
    for field in duration_fields:
        _nonnegative_number(step[field], f"{label} Host timing step {field}")
    launch_kinds = _sequence(
        step["launch_kinds"], f"{label} Host timing launch kinds")
    if any(kind not in LAUNCH_KINDS for kind in launch_kinds):
        raise ValueError(f"{label} Host timing launch kind is invalid")
    return step


def _validate_point(
    value: object,
    index: int,
    expected_scale: float,
    source_trace: Mapping[str, object],
    execution_provenance: Mapping[str, object],
) -> Mapping[str, object]:
    label = f"production Trace sweep point {index}"
    point = _mapping(value, label)
    _exact_fields(point, _POINT_FIELDS, label)
    scale = _positive_number(point["arrival_rate_scale"], f"{label} scale")
    if scale != expected_scale:
        raise ValueError(f"{label} scale disagrees with arrival_rate_scales")
    request_stream_sha256 = _sha256(
        point["request_stream_sha256"], f"{label} request stream SHA256")

    summary_metrics = _mapping(point["metrics"], f"{label} summary metrics")
    _exact_fields(summary_metrics, _SUMMARY_METRIC_FIELDS,
                  f"{label} summary metrics")
    result = _mapping(point["result"], f"{label} result")
    _required_fields(
        result,
        {
            "version",
            "backend",
            "execution_provenance",
            "execution_artifact",
            "request_source",
            "mtp_scenario",
            "metrics",
            "launch_policy",
            "hardware_ids",
            "performance_timing_status",
            "execution_mode",
            "serving_timing_mode",
            "host_timing",
        },
        f"{label} result",
    )
    if result["version"] != 3:
        raise ValueError(f"{label} result version must be 3")
    if result["backend"] != "vllm":
        raise ValueError(f"{label} result backend must be vllm")
    expected_timing_status = performance_timing_status(
        result["launch_policy"],
        tuple(_sequence(result["hardware_ids"], f"{label} hardware IDs")),
    )
    if result["performance_timing_status"] != expected_timing_status:
        raise ValueError(
            f"{label} performance timing status disagrees with launch policy "
            "and hardware IDs")
    status = _mapping(point["validation_status"], f"{label} validation status")
    _exact_fields(
        status,
        set(_EXPECTED_FIXED_STATUS) | {"performance_timing"},
        f"{label} validation status",
    )
    for field, expected in _EXPECTED_FIXED_STATUS.items():
        if status[field] != expected:
            raise ValueError(
                f"{label} validation status has invalid {field}")
    if status["performance_timing"] != expected_timing_status:
        raise ValueError(
            f"{label} validation status disagrees with result timing status")
    if result["execution_provenance"] != execution_provenance:
        raise ValueError(f"{label} execution provenance disagrees with sweep")
    if execution_provenance["performance_timing_status"] != (
            expected_timing_status):
        raise ValueError(
            f"{label} execution provenance timing status disagrees with "
            "result")
    _validate_execution_and_host_timing(result, label)
    try:
        mtp_scenario = normalize_mtp_scenario_manifest(
            result["mtp_scenario"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} MTP scenario is invalid: {error}") from error
    if _canonical_json_bytes(mtp_scenario) != _canonical_json_bytes(
            result["mtp_scenario"]):
        raise ValueError(f"{label} MTP scenario is not canonical")

    request_source = _mapping(result["request_source"], f"{label} request source")
    _required_fields(
        request_source, {"kind", "trace", "replay"}, f"{label} request source")
    if request_source["kind"] != "production_trace":
        raise ValueError(f"{label} request source must be production_trace")
    if request_source["trace"] != source_trace:
        raise ValueError(f"{label} source Trace provenance disagrees with sweep")
    replay = _mapping(request_source["replay"], f"{label} replay")
    _exact_fields(replay, _REPLAY_FIELDS, f"{label} replay")
    if replay["schema_version"] != 1:
        raise ValueError(f"{label} replay schema_version must be 1")
    if replay["kind"] != "arrival_rate_scale":
        raise ValueError(f"{label} replay kind must be arrival_rate_scale")
    if replay["arrival_transform"] != _ARRIVAL_TRANSFORM:
        raise ValueError(f"{label} replay arrival transform is unsupported")
    if replay["interpretation"] != "successful_path_performance_only":
        raise ValueError(f"{label} replay interpretation is unsupported")
    if _positive_number(replay["arrival_rate_scale"], f"{label} replay scale") != scale:
        raise ValueError(f"{label} replay scale disagrees with point")
    replay_sha256 = _sha256(
        replay["request_stream_sha256"], f"{label} replay request stream SHA256")
    if replay_sha256 != request_stream_sha256:
        raise ValueError(f"{label} request stream SHA256 disagrees with replay")

    result_metrics = _mapping(result["metrics"], f"{label} result metrics")
    _required_fields(
        result_metrics,
        _SUMMARY_METRIC_FIELDS | {"system_resources", "critical_path"},
        f"{label} result metrics",
    )
    for metric_name in _SUMMARY_METRIC_FIELDS:
        if summary_metrics[metric_name] != result_metrics[metric_name]:
            raise ValueError(
                f"{label} summary metrics disagree with nested result: "
                f"{metric_name}")
    for metric_name in (
            "request_throughput_per_s",
            "output_token_throughput_per_s",
            "token_throughput_per_s"):
        _nonnegative_number(result_metrics[metric_name], f"{label} {metric_name}")
    _validate_latency_distributions(result_metrics["latency_distributions"], label)
    _validate_slo_goodput(result_metrics["slo_goodput"], label)
    _validate_system_resources(result_metrics["system_resources"], label)
    _validate_critical_path(result_metrics["critical_path"], label)
    _validate_execution_artifact(
        result["execution_artifact"], label, expected_timing_status)
    _validate_host_kv_resource_binding(
        result["execution_artifact"],
        result_metrics["system_resources"],
        label,
    )
    return point


def _render_timing_scope(timing_status: object) -> list[str]:
    if timing_status == UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS:
        return [
            "> [!WARNING]",
            "> **UNSUPPORTED / DEVELOPMENT ONLY:** this run used A800 "
            "analytical launch. Do not use its timings for publication "
            "performance claims, accuracy validation, hardware comparison, "
            "or DSE conclusions.",
            "",
            "- Timing: **Unsupported / development-only** "
            f"(`{UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS}`).",
        ]
    if timing_status == PROJECTED_PERFORMANCE_TIMING_STATUS:
        return [
            "- Timing: **Projected** "
            f"(`{PROJECTED_PERFORMANCE_TIMING_STATUS}`).",
        ]
    raise AssertionError("validated performance timing status is unknown")


def _render_mtp_scenario(scenario: Mapping[str, object]) -> list[str]:
    policy = scenario["acceptance_policy"]
    if policy == "disabled":
        return ["- MTP acceptance: disabled."]
    if policy == "fixed_count":
        return [
            "- MTP acceptance: **Projected explicit assumption** "
            f"(`fixed_count`, accepted "
            f"`{scenario['fixed_accepted_tokens']}`/"
            f"`{scenario['num_speculative_tokens']}`, source "
            f"{_code(scenario['acceptance_source_id'])})."
        ]
    if policy == "maximum_entropy":
        assumption = _mapping(
            scenario["acceptance_assumption"],
            "MTP maximum-entropy acceptance assumption",
        )
        return [
            "- MTP acceptance: **Projected explicit assumption** "
            "(`maximum_entropy`, finite support "
            f"`[0,{assumption['proposal_depth']}]`, mean rate "
            f"`{assumption['mean_acceptance_rate']}`, seed "
            f"`{assumption['root_seed']}`, source "
            f"{_code(assumption['source_id'])}).",
            "- MTP assumption policy SHA256: "
            f"`{assumption['policy_descriptor_sha256']}`; duration "
            "observations: none.",
        ]
    if policy == "replay":
        replay = _mapping(
            scenario["acceptance_replay"], "MTP acceptance replay provenance")
        lines = [
            "- MTP acceptance: **Replayed external observation** "
            f"(`replay`, source {_code(scenario['acceptance_source_id'])}).",
            "- MTP replay artifact: "
            f"schema `{replay['schema_version']}`, config "
            f"{_code(replay['config_path'])}, SHA256 "
            f"`{replay['config_sha256']}`.",
            "- MTP replay producer: "
            f"{_code(replay['producer'])}; vLLM "
            f"{_code(replay['vllm_version'])} at "
            f"`{replay['vllm_source_commit']}`; package SHA256 "
            f"`{replay['vllm_package_manifest_sha256']}`.",
            "- MTP replay model config SHA256: "
            f"`{replay['model_config_sha256']}`.",
            "- MTP replay scope: pool "
            f"{_code(replay['pool_name'])}, role "
            f"{_code(replay['backend_role'])}, speculative tokens "
            f"`{replay['num_speculative_tokens']}`, Scheduler steps "
            f"`{replay['step_count']}`.",
        ]
        if replay["schema_version"] == 1:
            lines.append(
                "- MTP replay request-stream SHA256: "
                f"`{replay['request_stream_sha256']}`.")
        elif replay["schema_version"] == 2:
            lines.extend((
                "- MTP replay adapter: artifact schema "
                f"`{replay['artifact_schema_version']}`, contract "
                f"{_code(replay['adapter_contract'])}.",
                "- MTP replay ingress request-stream SHA256: "
                f"`{replay['ingress_request_stream_sha256']}`.",
                "- MTP replay decode-admission SHA256: "
                f"`{replay['decode_admission_contract_sha256']}`.",
            ))
        else:
            raise AssertionError(
                "validated MTP replay has an unknown schema version")
        return lines
    raise AssertionError("validated MTP scenario has an unknown policy")


def _validate_source_trace(value: object) -> Mapping[str, object]:
    trace = _mapping(value, "production Trace source provenance")
    _exact_fields(
        trace, _SOURCE_TRACE_FIELDS, "production Trace source provenance")
    if trace["schema_version"] != 1:
        raise ValueError("production Trace source schema_version must be 1")
    if trace["successful_only"] is not True:
        raise ValueError("production Trace report requires successful_only=True")
    if not isinstance(trace["source_path"], str) or not trace["source_path"]:
        raise ValueError("production Trace source path must be non-empty")
    _sha256(trace["source_sha256"], "production Trace source SHA256")
    _sha256(
        trace["normalized_request_stream_sha256"],
        "production Trace normalized request stream SHA256",
    )
    _mapping(trace["statistics"], "production Trace statistics")
    return trace


def _validate_execution_provenance(value: object) -> Mapping[str, object]:
    provenance = _mapping(value, "execution provenance")
    _exact_fields(
        provenance, _EXECUTION_PROVENANCE_FIELDS, "execution provenance")
    if provenance["schema_version"] != 3:
        raise ValueError("execution provenance schema_version must be 3")
    repository = _mapping(provenance["repository"], "repository provenance")
    _required_fields(
        repository,
        {"git_commit", "worktree_state", "git_diff_hash"},
        "repository provenance",
    )
    commit = repository["git_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError("execution provenance git_commit must be lowercase hex")
    if not isinstance(repository["worktree_state"], str):
        raise ValueError("execution provenance worktree_state must be a string")
    _sha256(
        repository["git_diff_hash"],
        "execution provenance git_diff_hash",
    )
    hardware_configs = _mapping(
        provenance["hardware_configs"], "hardware config provenance")
    _exact_fields(
        hardware_configs,
        {"prefill", "decode", "additional_colocated"},
        "hardware config provenance",
    )
    hardware_ids = _sequence(
        provenance["loaded_hardware_ids"], "loaded hardware IDs")
    if not hardware_ids or any(
            not isinstance(hardware_id, str) or not hardware_id
            for hardware_id in hardware_ids):
        raise ValueError("execution provenance loaded_hardware_ids are invalid")
    _mapping(provenance["runtime_inputs"], "runtime input provenance")
    launch_inputs = _mapping(
        provenance["launch_inputs"], "launch input provenance")
    _required_fields(
        launch_inputs, {"launch_policy", "launch_data"},
        "launch input provenance")
    _mapping(launch_inputs["launch_data"], "launch-data provenance")
    expected_timing_status = performance_timing_status(
        launch_inputs["launch_policy"], tuple(hardware_ids))
    if provenance["performance_timing_status"] != expected_timing_status:
        raise ValueError(
            "execution provenance performance timing status disagrees with "
            "launch policy and loaded hardware")
    _mapping(
        provenance["microbenchmark_inputs"],
        "microbenchmark input provenance",
    )
    if (not isinstance(provenance["prediction_method"], str)
            or not provenance["prediction_method"]):
        raise ValueError("execution provenance prediction_method is invalid")
    return provenance


def _validate_execution_artifact(
    value: object,
    point_label: str,
    expected_timing_status: str,
) -> None:
    artifact = _mapping(value, f"{point_label} execution artifact")
    _exact_fields(artifact, EXECUTION_ARTIFACT_FIELDS, f"{point_label} execution artifact")
    if artifact["schema_version"] not in (5, 6, 7, 8):
        raise ValueError(
            f"{point_label} execution artifact schema_version must be 5, 6, "
            "7, or 8")
    modeled_kv_offload = _mapping(
        artifact["modeled_kv_offload"],
        f"{point_label} modeled KV offload",
    )
    expected_offload_schema = {
        5: 2,
        6: 3,
        7: 4,
        8: modeled_kv_offload["schema_version"],
    }[artifact["schema_version"]]
    if (artifact["schema_version"] == 8
            and expected_offload_schema not in (2, 4)):
        raise ValueError(
            f"{point_label} compact execution artifact has unsupported "
            "modeled KV offload schema")
    if modeled_kv_offload["schema_version"] != expected_offload_schema:
        raise ValueError(
            f"{point_label} modeled KV offload schema_version disagrees with "
            "the execution artifact")
    _validate_modeled_kv_offload_artifact(
        artifact, modeled_kv_offload, point_label)
    if artifact["performance_timing_status"] != expected_timing_status:
        raise ValueError(
            f"{point_label} execution artifact performance timing status "
            "disagrees with result")
    if artifact["hash_method"] != "canonical_json_sort_keys_compact_utf8_v1":
        raise ValueError(
            f"{point_label} execution artifact hash_method is unsupported")
    expected = _sha256(
        artifact["artifact_sha256"],
        f"{point_label} execution artifact SHA256",
    )
    body = {
        key: artifact[key] for key in artifact if key != "artifact_sha256"
    }
    try:
        serialized = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{point_label} execution artifact is not canonical JSON") from error
    actual = hashlib.sha256(serialized).hexdigest()
    if actual != expected:
        raise ValueError(f"{point_label} execution artifact SHA256 is invalid")


def _validate_modeled_kv_offload_artifact(
    artifact: Mapping[str, object],
    modeled: Mapping[str, object],
    point_label: str,
) -> None:
    label = f"{point_label} modeled KV offload"
    counts = _mapping(
        artifact["counts"], f"{point_label} execution artifact counts")
    _required_fields(
        counts,
        {"modeled_kv_offload_transfers"},
        f"{point_label} execution artifact counts",
    )
    expected_transfer_count = _nonnegative_integer(
        counts["modeled_kv_offload_transfers"],
        f"{point_label} modeled KV offload transfer count",
    )

    if (artifact["schema_version"] in (7, 8)
            and modeled["schema_version"] == 4):
        _validate_modeled_kv_offload_v4(
            artifact,
            modeled,
            expected_transfer_count,
            label,
        )
        return

    if (artifact["schema_version"] in (5, 8)
            and modeled["schema_version"] == 2):
        _required_fields(modeled, {"availability"}, label)
        if modeled["availability"] == "not_enabled":
            _exact_fields(
                modeled,
                {
                    "schema_version", "availability", "intent_source",
                    "transfers", "final_tier_state",
                },
                label,
            )
            transfers = _sequence(
                modeled["transfers"], f"{label} transfers")
            if len(transfers) != expected_transfer_count:
                raise ValueError(
                    f"{label} transfers disagree with execution artifact "
                    "counts")
            if transfers or modeled["final_tier_state"] is not None:
                raise ValueError(
                    f"{label} disabled contract cannot contain transfers or "
                    "tier state")
            return
        _exact_fields(
            modeled,
            {
                "schema_version", "availability", "mode",
                "validation_status", "intent_source", "scope", "transfers",
                "final_tier_state",
            },
            label,
        )
        transfers = _sequence(
            modeled["transfers"], f"{label} transfers")
        if len(transfers) != expected_transfer_count:
            raise ValueError(
                f"{label} transfers disagree with execution artifact counts")
        if modeled["availability"] != "available_modeled_serial":
            raise ValueError(f"{label} v2 availability is invalid")
        for index, raw_transfer in enumerate(transfers):
            transfer = _mapping(
                raw_transfer, f"{label} transfer {index}")
            _exact_fields(
                transfer,
                {
                    "step_id", "ready_event", "intent", "node_id",
                    "call_id", "relative_interval", "absolute_interval",
                    "prediction_total_s", "completion",
                },
                f"{label} transfer {index}",
            )
        _mapping(modeled["final_tier_state"], f"{label} final tier state")
        return
    if artifact["schema_version"] != 6:
        raise AssertionError("validated artifact schema version vanished")

    _exact_fields(
        modeled,
        {
            "schema_version", "availability", "mode", "validation_status",
            "intent_source", "ledger_semantics", "physical_transfer_count",
            "scope", "transfers", "final_tier_state",
        },
        label,
    )
    transfers = _sequence(modeled["transfers"], f"{label} transfers")
    if len(transfers) != expected_transfer_count:
        raise ValueError(
            f"{label} transfers disagree with execution artifact counts")
    if modeled["availability"] != "available_modeled_rank_local":
        raise ValueError(f"{label} v3 availability is invalid")
    if modeled["mode"] != "modeled_host_kv_rank_local_lockstep":
        raise ValueError(f"{label} v3 mode is invalid")
    if modeled["ledger_semantics"] != (
            "logical_request_blocks_with_lockstep_per_rank_capacity"):
        raise ValueError(f"{label} v3 ledger semantics are invalid")
    if not transfers:
        raise ValueError(f"{label} v3 requires logical transfers")

    physical_count = 0
    for index, raw_transfer in enumerate(transfers):
        transfer_label = f"{label} transfer {index}"
        transfer = _mapping(raw_transfer, transfer_label)
        _exact_fields(
            transfer,
            {
                "step_id", "ready_event", "intent", "logical_completion",
                "physical_transfers",
            },
            transfer_label,
        )
        intent = _mapping(transfer["intent"], f"{transfer_label} intent")
        _required_fields(
            intent,
            {
                "ready_event", "action", "bytes", "source_tier",
                "destination_tier",
            },
            f"{transfer_label} intent",
        )
        logical = _mapping(
            transfer["logical_completion"],
            f"{transfer_label} logical completion",
        )
        _exact_fields(
            logical,
            {
                "ready_event", "action", "bytes_per_rank", "source_tier",
                "destination_tier", "link", "physical_transfer_count",
                "relative_interval", "absolute_interval",
            },
            f"{transfer_label} logical completion",
        )
        bytes_per_rank = _nonnegative_integer(
            logical["bytes_per_rank"],
            f"{transfer_label} bytes per rank",
        )
        if bytes_per_rank == 0:
            raise ValueError(f"{transfer_label} bytes per rank must be positive")
        if (transfer["ready_event"] != intent["ready_event"]
                or transfer["ready_event"] != logical["ready_event"]
                or intent["action"] != logical["action"]
                or intent["bytes"] != bytes_per_rank
                or intent["source_tier"] != logical["source_tier"]
                or intent["destination_tier"]
                != logical["destination_tier"]):
            raise ValueError(
                f"{transfer_label} logical provenance is inconsistent")
        physical_transfers = _sequence(
            transfer["physical_transfers"],
            f"{transfer_label} physical transfers",
        )
        if len(physical_transfers) != 2:
            raise ValueError(
                f"{transfer_label} requires exactly two physical transfers")
        if logical["physical_transfer_count"] != 2:
            raise ValueError(
                f"{transfer_label} logical physical count must be two")
        ranks = []
        for physical_index, raw_physical in enumerate(physical_transfers):
            physical_label = (
                f"{transfer_label} physical transfer {physical_index}")
            physical = _mapping(raw_physical, physical_label)
            _exact_fields(
                physical,
                {
                    "rank", "pool_name", "node_id", "call_id",
                    "relative_interval", "absolute_interval",
                    "prediction_total_s", "completion",
                },
                physical_label,
            )
            completion = _mapping(
                physical["completion"], f"{physical_label} completion")
            _exact_fields(
                completion,
                {
                    "node_id", "call_id", "pool_name", "rank", "bytes",
                    "link", "start_time_s", "end_time_s",
                },
                f"{physical_label} completion",
            )
            rank = _nonnegative_integer(
                physical["rank"], f"{physical_label} rank")
            ranks.append(rank)
            if (completion["rank"] != rank
                    or completion["pool_name"] != physical["pool_name"]
                    or completion["node_id"] != physical["node_id"]
                    or completion["call_id"] != physical["call_id"]
                    or completion["bytes"] != bytes_per_rank
                    or completion["link"] != logical["link"]):
                raise ValueError(
                    f"{physical_label} completion provenance is inconsistent")
            interval = _mapping(
                physical["relative_interval"],
                f"{physical_label} relative interval",
            )
            _exact_fields(
                interval, {"start_time_s", "end_time_s"},
                f"{physical_label} relative interval")
            start = _nonnegative_number(
                interval["start_time_s"], f"{physical_label} start")
            end = _nonnegative_number(
                interval["end_time_s"], f"{physical_label} end")
            prediction = _positive_number(
                physical["prediction_total_s"],
                f"{physical_label} prediction")
            if (end <= start
                    or not math.isclose(
                        end - start, prediction,
                        rel_tol=1e-12, abs_tol=1e-15)
                    or completion["start_time_s"] != interval["start_time_s"]
                    or completion["end_time_s"] != interval["end_time_s"]):
                raise ValueError(
                    f"{physical_label} interval/prediction is inconsistent")
        if tuple(ranks) != (0, 1):
            raise ValueError(
                f"{transfer_label} physical ranks must be exactly (0, 1)")
        physical_count += len(physical_transfers)
    declared_physical_count = _nonnegative_integer(
        modeled["physical_transfer_count"],
        f"{label} physical transfer count",
    )
    if declared_physical_count != physical_count:
        raise ValueError(f"{label} physical transfer count is inconsistent")
    _mapping(modeled["final_tier_state"], f"{label} final tier state")


def _validate_modeled_kv_offload_v4(
    artifact: Mapping[str, object],
    modeled: Mapping[str, object],
    expected_transfer_count: int,
    label: str,
) -> None:
    _exact_fields(
        modeled,
        {
            "schema_version", "availability", "mode", "validation_status",
            "intent_source", "ledger_semantics", "physical_transfer_count",
            "scope", "transfers", "final_tier_state",
        },
        label,
    )
    if modeled["schema_version"] != 4:
        raise ValueError(f"{label} schema_version must be 4")
    schedule_trace = _mapping(
        artifact["schedule_trace"], f"{label} schedule trace")
    schedule_entries = _sequence(
        schedule_trace["entries"], f"{label} schedule entries")
    schedule_entries_by_id = {}
    for index, raw_entry in enumerate(schedule_entries):
        entry = _mapping(raw_entry, f"{label} schedule entry {index}")
        _required_fields(
            entry, {"step_id", "pool_name", "start_time_s", "end_time_s"},
            f"{label} schedule entry {index}")
        step_id = entry["step_id"]
        if step_id in schedule_entries_by_id:
            raise ValueError(f"{label} has duplicate schedule step IDs")
        schedule_entries_by_id[step_id] = entry
    steps = _sequence(artifact["steps"], f"{label} artifact steps")
    steps_by_id = {}
    for index, raw_step in enumerate(steps):
        step = _mapping(raw_step, f"{label} artifact step {index}")
        _required_fields(
            step,
            {
                "step_id", "pool_name", "schedule_interval",
                "workload_graph", "kernel_plan", "predictions",
                "execution_timeline",
            },
            f"{label} artifact step {index}",
        )
        step_id = step["step_id"]
        if step_id in steps_by_id:
            raise ValueError(f"{label} has duplicate artifact step IDs")
        if step_id not in schedule_entries_by_id:
            raise ValueError(f"{label} artifact step lacks a schedule entry")
        entry = schedule_entries_by_id[step_id]
        schedule_interval = _mapping(
            step["schedule_interval"],
            f"{label} step {step_id} schedule interval",
        )
        _exact_fields(
            schedule_interval, {"start_time_s", "end_time_s"},
            f"{label} step {step_id} schedule interval")
        if (step["pool_name"] != entry["pool_name"]
                or schedule_interval["start_time_s"] != entry["start_time_s"]
                or schedule_interval["end_time_s"] != entry["end_time_s"]):
            raise ValueError(
                f"{label} artifact step disagrees with its schedule entry")
        steps_by_id[step_id] = step
    if set(steps_by_id) != set(schedule_entries_by_id):
        raise ValueError(f"{label} step/schedule coverage is inconsistent")
    availability = modeled["availability"]
    if availability == "available_modeled_serial":
        expected_mode = "modeled_host_kv_serial"
        expected_ledger = (
            "logical_request_blocks_with_single_rank_capacity")
        expected_physical_per_logical = 1
        expected_multi_rank = "unsupported"
    elif availability == "available_modeled_rank_local":
        expected_mode = "modeled_host_kv_rank_local_lockstep"
        expected_ledger = (
            "logical_request_blocks_with_lockstep_per_rank_capacity")
        expected_physical_per_logical = 2
        expected_multi_rank = "validated_colocated_qwen3_tp2_lockstep"
    else:
        raise ValueError(f"{label} availability is invalid")
    if (modeled["mode"] != expected_mode
            or modeled["ledger_semantics"] != expected_ledger):
        raise ValueError(f"{label} mode or ledger semantics are invalid")
    scope = _mapping(modeled["scope"], f"{label} scope")
    _exact_fields(
        scope,
        {
            "vllm_native_offload", "ramulator", "host_dram_timing",
            "host_dram_queue", "host_link_utilization", "prefetch",
            "compute_overlap", "multi_pool", "multi_rank",
        },
        f"{label} scope",
    )
    if scope["host_link_utilization"] != (
            "available_in_system_resource_manifest_v5_"
            "host_kv_link_utilization_per_rank"):
        raise ValueError(
            f"{label} does not bind SystemResourceManifest v5 Host-KV "
            "utilization")
    if scope["multi_rank"] != expected_multi_rank:
        raise ValueError(f"{label} multi-rank scope is inconsistent")

    transfers = _sequence(modeled["transfers"], f"{label} transfers")
    if len(transfers) != expected_transfer_count or not transfers:
        raise ValueError(
            f"{label} transfers disagree with execution artifact counts")
    physical_count = 0
    for index, raw_transfer in enumerate(transfers):
        transfer_label = f"{label} transfer {index}"
        transfer = _mapping(raw_transfer, transfer_label)
        _exact_fields(
            transfer,
            {
                "step_id", "ready_event", "intent", "logical_completion",
                "physical_transfers",
            },
            transfer_label,
        )
        intent = _mapping(transfer["intent"], f"{transfer_label} intent")
        _required_fields(
            intent,
            {
                "ready_event", "action", "bytes", "source_tier",
                "destination_tier", "tier",
            },
            f"{transfer_label} intent",
        )
        logical = _mapping(
            transfer["logical_completion"],
            f"{transfer_label} logical completion",
        )
        _exact_fields(
            logical,
            {
                "ready_event", "action", "bytes_per_rank", "source_tier",
                "destination_tier", "link", "physical_transfer_count",
                "relative_interval", "absolute_interval",
            },
            f"{transfer_label} logical completion",
        )
        bytes_per_rank = _nonnegative_integer(
            logical["bytes_per_rank"],
            f"{transfer_label} bytes per rank",
        )
        if bytes_per_rank == 0:
            raise ValueError(f"{transfer_label} bytes per rank must be positive")
        if (transfer["ready_event"] != intent["ready_event"]
                or transfer["ready_event"] != logical["ready_event"]
                or intent["action"] not in ("evict", "load")
                or intent["action"] != logical["action"]
                or intent["bytes"] != bytes_per_rank
                or intent["source_tier"] != logical["source_tier"]
                or intent["destination_tier"]
                != logical["destination_tier"]):
            raise ValueError(
                f"{transfer_label} logical provenance is inconsistent")
        physical_transfers = _sequence(
            transfer["physical_transfers"],
            f"{transfer_label} physical transfers",
        )
        if len(physical_transfers) != expected_physical_per_logical:
            raise ValueError(
                f"{transfer_label} physical transfer count is invalid")
        if logical["physical_transfer_count"] != len(physical_transfers):
            raise ValueError(
                f"{transfer_label} logical physical count is inconsistent")

        step_id = transfer["step_id"]
        if step_id not in steps_by_id:
            raise ValueError(f"{transfer_label} references an unknown step")
        step = steps_by_id[step_id]
        step_origin_s = _nonnegative_number(
            schedule_entries_by_id[step_id]["start_time_s"],
            f"{transfer_label} step origin",
        )
        nodes = _sequence(
            _mapping(
                step["workload_graph"],
                f"{transfer_label} workload graph",
            )["nodes"],
            f"{transfer_label} workload nodes",
        )
        nodes_by_id = {}
        for raw_node in nodes:
            node = _mapping(raw_node, f"{transfer_label} workload node")
            node_id = node["node_id"]
            if node_id in nodes_by_id:
                raise ValueError(
                    f"{transfer_label} has duplicate workload node IDs")
            nodes_by_id[node_id] = node
        calls = _sequence(
            _mapping(
                step["kernel_plan"], f"{transfer_label} kernel plan",
            )["calls"],
            f"{transfer_label} kernel calls",
        )
        calls_by_id = {}
        for raw_call in calls:
            call = _mapping(raw_call, f"{transfer_label} kernel call")
            call_id = call["call_id"]
            if call_id in calls_by_id:
                raise ValueError(
                    f"{transfer_label} has duplicate kernel call IDs")
            calls_by_id[call_id] = call
        predictions = _sequence(
            step["predictions"], f"{transfer_label} predictions")
        predictions_by_id = {}
        for raw_prediction in predictions:
            prediction = _mapping(
                raw_prediction, f"{transfer_label} prediction")
            call_id = prediction["call_id"]
            if call_id in predictions_by_id:
                raise ValueError(
                    f"{transfer_label} has duplicate prediction IDs")
            predictions_by_id[call_id] = prediction
        timeline = _mapping(
            step["execution_timeline"], f"{transfer_label} timeline")
        if timeline["origin_time_s"] != step_origin_s:
            raise ValueError(
                f"{transfer_label} timeline origin disagrees with schedule")
        intervals = _sequence(
            timeline["intervals"], f"{transfer_label} intervals")
        intervals_by_id = {}
        for raw_interval in intervals:
            interval = _mapping(
                raw_interval, f"{transfer_label} interval")
            call_id = interval["call_id"]
            if call_id in intervals_by_id:
                raise ValueError(
                    f"{transfer_label} has duplicate interval IDs")
            intervals_by_id[call_id] = interval

        ranks = []
        pools = []
        relative_intervals = []
        absolute_intervals = []
        for physical_index, raw_physical in enumerate(physical_transfers):
            physical_label = (
                f"{transfer_label} physical transfer {physical_index}")
            physical = _mapping(raw_physical, physical_label)
            _exact_fields(
                physical,
                {
                    "resource_id", "rank", "pool_name", "node_id",
                    "call_id", "relative_interval", "absolute_interval",
                    "prediction_total_s", "completion",
                },
                physical_label,
            )
            completion = _mapping(
                physical["completion"], f"{physical_label} completion")
            _exact_fields(
                completion,
                {
                    "node_id", "call_id", "pool_name", "rank", "bytes",
                    "link", "start_time_s", "end_time_s",
                },
                f"{physical_label} completion",
            )
            rank = _nonnegative_integer(
                physical["rank"], f"{physical_label} rank")
            ranks.append(rank)
            pool_name = physical["pool_name"]
            if (not isinstance(pool_name, str) or not pool_name
                    or not isinstance(logical["link"], str)
                    or not logical["link"]):
                raise ValueError(
                    f"{physical_label} pool/link must be non-empty")
            pools.append(pool_name)
            expected_resource_id = (
                f"pool:{pool_name}:rank:{rank}:"
                f"{logical['link']}")
            if (physical["resource_id"] != expected_resource_id
                    or completion["rank"] != rank
                    or completion["pool_name"] != physical["pool_name"]
                    or completion["node_id"] != physical["node_id"]
                    or completion["call_id"] != physical["call_id"]
                    or completion["bytes"] != bytes_per_rank
                    or completion["link"] != logical["link"]):
                raise ValueError(
                    f"{physical_label} completion/resource provenance is "
                    "inconsistent")
            node_id = physical["node_id"]
            call_id = physical["call_id"]
            if (node_id not in nodes_by_id
                    or call_id not in calls_by_id
                    or call_id not in predictions_by_id
                    or call_id not in intervals_by_id):
                raise ValueError(
                    f"{physical_label} lacks a canonical node/call/"
                    "prediction/interval")
            node = nodes_by_id[node_id]
            node_attributes = _mapping(
                node["attributes"], f"{physical_label} node attributes")
            call = calls_by_id[call_id]
            spec = _mapping(call["kernel_spec"], f"{physical_label} spec")
            transfer_config = _mapping(
                spec["transfer"], f"{physical_label} transfer config")
            prediction_record = predictions_by_id[call_id]
            interval_record = intervals_by_id[call_id]
            if (node["kind"] != "kv_offload"
                    or node["pool_name"] != pool_name
                    or node["rank"] != rank
                    or node_attributes["ready_event"]
                    != transfer["ready_event"]
                    or node_attributes["bytes"] != bytes_per_rank
                    or node_attributes["direction"] != logical["action"]
                    or node_attributes["source_tier"]
                    != logical["source_tier"]
                    or node_attributes["destination_tier"]
                    != logical["destination_tier"]
                    or node_attributes["link"] != logical["link"]
                    or call["pool_name"] != pool_name
                    or call["rank"] != rank
                    or call["resource"] != logical["link"]
                    or tuple(_sequence(
                        call["source_node_ids"],
                        f"{physical_label} source node IDs")) != (node_id,)
                    or spec["kind"] != "KV_OFFLOAD"
                    or transfer_config["algorithm"] != "dma"
                    or transfer_config["bytes"] != bytes_per_rank
                    or transfer_config["direction"] != logical["action"]
                    or transfer_config["link"] != logical["link"]
                    or transfer_config["tier"] != intent["tier"]
                    or prediction_record["t_total_s"]
                    != physical["prediction_total_s"]
                    or interval_record["pool_name"] != pool_name
                    or interval_record["rank"] != rank
                    or interval_record["resource"] != logical["link"]
                    or tuple(_sequence(
                        interval_record["reserved_resource_ids"],
                        f"{physical_label} reserved resources"))
                    != (physical["resource_id"],)):
                raise ValueError(
                    f"{physical_label} disagrees with canonical execution "
                    "components")
            relative = _mapping(
                physical["relative_interval"],
                f"{physical_label} relative interval",
            )
            absolute = _mapping(
                physical["absolute_interval"],
                f"{physical_label} absolute interval",
            )
            _exact_fields(
                relative, {"start_time_s", "end_time_s"},
                f"{physical_label} relative interval")
            _exact_fields(
                absolute, {"start_time_s", "end_time_s"},
                f"{physical_label} absolute interval")
            relative_start = _nonnegative_number(
                relative["start_time_s"], f"{physical_label} relative start")
            relative_end = _nonnegative_number(
                relative["end_time_s"], f"{physical_label} relative end")
            absolute_start = _nonnegative_number(
                absolute["start_time_s"], f"{physical_label} absolute start")
            absolute_end = _nonnegative_number(
                absolute["end_time_s"], f"{physical_label} absolute end")
            prediction = _positive_number(
                physical["prediction_total_s"],
                f"{physical_label} prediction")
            if (relative_end <= relative_start
                    or not math.isclose(
                        relative_end - relative_start,
                        prediction,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        absolute_end - absolute_start,
                        prediction,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        absolute_start - relative_start,
                        step_origin_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        absolute_end - relative_end,
                        step_origin_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or completion["start_time_s"] != relative_start
                    or completion["end_time_s"] != relative_end
                    or interval_record["start_time_s"] != relative_start
                    or interval_record["end_time_s"] != relative_end
                    or interval_record["absolute_start_time_s"]
                    != absolute_start
                    or interval_record["absolute_end_time_s"]
                    != absolute_end):
                raise ValueError(
                    f"{physical_label} interval/prediction is inconsistent")
            relative_intervals.append((relative_start, relative_end))
            absolute_intervals.append((absolute_start, absolute_end))
        if (ranks != sorted(ranks)
                or len(ranks) != len(set(ranks))
                or len(set(pools)) != 1):
            raise ValueError(
                f"{transfer_label} physical ranks/pools are inconsistent")
        if (expected_physical_per_logical == 1 and len(ranks) != 1
                or expected_physical_per_logical == 2
                and tuple(ranks) != (0, 1)):
            raise ValueError(
                f"{transfer_label} physical ranks are outside validated "
                "scope")

        logical_relative = _mapping(
            logical["relative_interval"],
            f"{transfer_label} logical relative interval",
        )
        logical_absolute = _mapping(
            logical["absolute_interval"],
            f"{transfer_label} logical absolute interval",
        )
        _exact_fields(
            logical_relative, {"start_time_s", "end_time_s"},
            f"{transfer_label} logical relative interval")
        _exact_fields(
            logical_absolute, {"start_time_s", "end_time_s"},
            f"{transfer_label} logical absolute interval")
        if (logical_relative["start_time_s"]
                != min(item[0] for item in relative_intervals)
                or logical_relative["end_time_s"]
                != max(item[1] for item in relative_intervals)
                or logical_absolute["start_time_s"]
                != min(item[0] for item in absolute_intervals)
                or logical_absolute["end_time_s"]
                != max(item[1] for item in absolute_intervals)
                or not math.isclose(
                    logical_absolute["start_time_s"]
                    - logical_relative["start_time_s"],
                    step_origin_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    logical_absolute["end_time_s"]
                    - logical_relative["end_time_s"],
                    step_origin_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{transfer_label} logical interval does not cover its "
                "physical transfers")
        physical_count += len(physical_transfers)

    declared_physical_count = _nonnegative_integer(
        modeled["physical_transfer_count"],
        f"{label} physical transfer count",
    )
    if declared_physical_count != physical_count:
        raise ValueError(f"{label} physical transfer count is inconsistent")
    _mapping(modeled["final_tier_state"], f"{label} final tier state")


def _validate_latency_distributions(value: object, point_label: str) -> None:
    distributions = _mapping(value, f"{point_label} latency distributions")
    expected_fields = {
        "schema_version",
        "quantile_representation",
        "percentile_method",
        "cdf_representation",
        "cdf_method",
        "ttft_s",
        "tpot_s",
        "latency_s",
    }
    _exact_fields(
        distributions, expected_fields, f"{point_label} latency distributions")
    if distributions["schema_version"] != 2:
        raise ValueError(f"{point_label} latency distribution schema must be 2")
    for metric_name in ("ttft_s", "tpot_s", "latency_s"):
        distribution = _mapping(
            distributions[metric_name], f"{point_label} {metric_name} distribution")
        _exact_fields(
            distribution,
            {
                "availability",
                "eligible_request_count",
                "excluded_request_count",
                "exclusion_reason",
                "quantiles",
                "empirical_cdf",
            },
            f"{point_label} {metric_name} distribution",
        )
        eligible = _nonnegative_integer(
            distribution["eligible_request_count"],
            f"{point_label} {metric_name} eligible count",
        )
        _nonnegative_integer(
            distribution["excluded_request_count"],
            f"{point_label} {metric_name} excluded count",
        )
        quantiles = _mapping(
            distribution["quantiles"], f"{point_label} {metric_name} quantiles")
        _exact_fields(
            quantiles, {"percentiles", "values_s"},
            f"{point_label} {metric_name} quantiles")
        percentiles = tuple(_sequence(
            quantiles["percentiles"], f"{point_label} {metric_name} percentiles"))
        values = tuple(_sequence(
            quantiles["values_s"], f"{point_label} {metric_name} quantile values"))
        cdf = _mapping(
            distribution["empirical_cdf"], f"{point_label} {metric_name} CDF")
        _exact_fields(
            cdf, {"values_s", "cumulative_probabilities"},
            f"{point_label} {metric_name} CDF")
        cdf_values = tuple(_sequence(
            cdf["values_s"], f"{point_label} {metric_name} CDF values"))
        probabilities = tuple(_sequence(
            cdf["cumulative_probabilities"],
            f"{point_label} {metric_name} CDF probabilities",
        ))
        availability = distribution["availability"]
        if availability == "available":
            if eligible == 0:
                raise ValueError(f"{point_label} available {metric_name} is empty")
            if percentiles != tuple(range(101)) or len(values) != 101:
                raise ValueError(
                    f"{point_label} {metric_name} must contain P0-P100")
            _nondecreasing_nonnegative(values, f"{point_label} {metric_name} quantiles")
            if len(cdf_values) != len(probabilities) or not cdf_values:
                raise ValueError(f"{point_label} {metric_name} CDF is incomplete")
            _nondecreasing_nonnegative(cdf_values, f"{point_label} {metric_name} CDF")
            previous = 0.0
            for probability in probabilities:
                normalized = _nonnegative_number(
                    probability, f"{point_label} {metric_name} CDF probability")
                if normalized <= previous or normalized > 1:
                    raise ValueError(
                        f"{point_label} {metric_name} CDF probabilities are invalid")
                previous = normalized
            if previous != 1.0:
                raise ValueError(f"{point_label} {metric_name} CDF must end at 1")
        elif availability == "unavailable_no_eligible_requests":
            if eligible != 0 or percentiles or values or cdf_values or probabilities:
                raise ValueError(
                    f"{point_label} unavailable {metric_name} contains values")
        else:
            raise ValueError(
                f"{point_label} unknown {metric_name} availability: {availability!r}")


def _validate_slo_goodput(value: object, point_label: str) -> None:
    slo = _mapping(value, f"{point_label} SLO goodput")
    _exact_fields(
        slo,
        {
            "schema_version",
            "availability",
            "contract",
            "population",
            "violation_counts",
            "request_goodput_per_s",
            "token_goodput_per_s",
        },
        f"{point_label} SLO goodput",
    )
    if slo["schema_version"] != 1:
        raise ValueError(f"{point_label} SLO goodput schema_version must be 1")
    population = _mapping(
        slo["population"], f"{point_label} SLO population")
    _exact_fields(
        population,
        {
            "total_request_count",
            "eligible_request_count",
            "excluded_request_count",
            "slo_met_request_count",
            "violated_request_count",
            "eligible_token_count",
            "slo_met_token_count",
        },
        f"{point_label} SLO population",
    )
    availability = slo["availability"]
    if availability == "unavailable_no_slo_contract":
        if (slo["contract"] is not None
                or slo["violation_counts"] is not None
                or slo["request_goodput_per_s"] is not None
                or slo["token_goodput_per_s"] is not None):
            raise ValueError(
                f"{point_label} unavailable SLO goodput contains results")
    elif availability == "available":
        contract = _mapping(slo["contract"], f"{point_label} SLO contract")
        _slo_contract(contract, point_label)
        violations = _mapping(
            slo["violation_counts"], f"{point_label} SLO violations")
        _exact_fields(
            violations,
            {"ttft_only", "tpot_only", "ttft_and_tpot"},
            f"{point_label} SLO violations",
        )
        _nonnegative_number(
            slo["request_goodput_per_s"], f"{point_label} request SLO goodput")
        _nonnegative_number(
            slo["token_goodput_per_s"], f"{point_label} token SLO goodput")
    else:
        raise ValueError(f"{point_label} unknown SLO availability: {availability!r}")


def _slo_contract(
    contract: Mapping[str, object],
    point_label: str,
) -> SLOContract:
    expected_fields = {
        "schema_version",
        "contract_id",
        "ttft_threshold_s",
        "tpot_threshold_s",
        "comparison",
        "threshold_composition",
        "request_eligibility",
        "single_output_tpot",
        "token_basis",
        "duration_basis",
        "source_path",
        "source_sha256",
    }
    _exact_fields(contract, expected_fields, f"{point_label} SLO contract")
    return SLOContract(
        schema_version=contract["schema_version"],
        contract_id=contract["contract_id"],
        ttft_threshold_s=contract["ttft_threshold_s"],
        tpot_threshold_s=contract["tpot_threshold_s"],
        comparison=contract["comparison"],
        threshold_composition=contract["threshold_composition"],
        request_eligibility=contract["request_eligibility"],
        single_output_tpot=contract["single_output_tpot"],
        token_basis=contract["token_basis"],
        duration_basis=contract["duration_basis"],
        source_path=contract["source_path"],
        source_sha256=contract["source_sha256"],
    )


def _validate_system_resources(value: object, point_label: str) -> None:
    resources = _mapping(value, f"{point_label} system resources")
    _required_fields(
        resources, {"schema_version"}, f"{point_label} system resources")
    common_fields = {
        "schema_version",
        "observation_window",
        "worker_pool_activity_basis",
        "link_activity_basis",
        "communication_resource_activity_basis",
        "worker_pool_utilization",
        "link_utilization",
        "communication_resource_utilization",
        "router_queue",
        "scheduler_queue",
    }
    if resources["schema_version"] == 4:
        _exact_fields(
            resources, common_fields, f"{point_label} system resources")
    elif resources["schema_version"] == 5:
        _exact_fields(
            resources,
            common_fields | {
                "host_kv_link_activity_basis",
                "host_kv_link_utilization",
            },
            f"{point_label} system resources",
        )
    else:
        raise ValueError(
            f"{point_label} system resource schema_version must be 4 or 5")
    window = _mapping(
        resources["observation_window"], f"{point_label} observation window")
    _required_fields(window, {"basis", "duration_s"},
                     f"{point_label} observation window")
    _positive_number(window["duration_s"], f"{point_label} observation duration")
    for field in (
            "worker_pool_utilization",
            "link_utilization",
            "communication_resource_utilization",
            "router_queue",
            "scheduler_queue"):
        _mapping(resources[field], f"{point_label} {field}")
    if resources["schema_version"] == 5:
        _validate_host_kv_link_resources(resources, point_label)
    worker_pools = _mapping(
        resources["worker_pool_utilization"],
        f"{point_label} worker-pool utilization",
    )
    if not worker_pools:
        raise ValueError(f"{point_label} has no worker-pool utilization")
    for pool_name, record_value in worker_pools.items():
        record = _mapping(
            record_value, f"{point_label} worker pool {pool_name}")
        _exact_fields(
            record,
            {"kernel_interval_count", "busy_time_s", "idle_time_s", "utilization"},
            f"{point_label} worker pool {pool_name}",
        )
        _nonnegative_integer(
            record["kernel_interval_count"],
            f"{point_label} worker pool {pool_name} interval count",
        )
        for field in ("busy_time_s", "idle_time_s", "utilization"):
            _nonnegative_number(
                record[field], f"{point_label} worker pool {pool_name} {field}")
    router = _mapping(resources["router_queue"], f"{point_label} router queue")
    _required_fields(
        router, {"availability", "overall", "by_stage"},
        f"{point_label} router queue")
    if router["availability"] == "modeled_serial_dispatch_resource":
        _exact_fields(
            router,
            {
                "availability",
                "timing_semantics",
                "resource_id",
                "configured_dispatch_service_time_s",
                "overall",
                "by_stage",
                "dispatch_service",
            },
            f"{point_label} router queue",
        )
    elif router["availability"] == "not_applicable_single_pool_without_router":
        _exact_fields(
            router, {"availability", "overall", "by_stage"},
            f"{point_label} router queue")
    else:
        raise ValueError(
            f"{point_label} router queue availability is unsupported")
    scheduler = _mapping(
        resources["scheduler_queue"], f"{point_label} scheduler queue")
    _required_fields(
        scheduler,
        {"availability", "admission_basis", "overall", "by_stage", "by_pool"},
        f"{point_label} scheduler queue",
    )
    _exact_fields(
        scheduler,
        {"availability", "admission_basis", "overall", "by_stage", "by_pool"},
        f"{point_label} scheduler queue",
    )


def _validate_host_kv_link_resources(
    resources: Mapping[str, object],
    point_label: str,
) -> None:
    if resources["host_kv_link_activity_basis"] != (
            "per_rank_resource_union_of_canonical_kv_offload_physical_"
            "completion_intervals"):
        raise ValueError(
            f"{point_label} Host-KV link activity basis is unsupported")
    host_resources = _mapping(
        resources["host_kv_link_utilization"],
        f"{point_label} Host-KV link utilization",
    )
    window = _mapping(
        resources["observation_window"],
        f"{point_label} observation window",
    )
    duration_s = _positive_number(
        window["duration_s"], f"{point_label} observation duration")
    for resource_id, raw_record in host_resources.items():
        label = f"{point_label} Host-KV resource {resource_id}"
        record = _mapping(raw_record, label)
        _exact_fields(
            record,
            {
                "pool_name", "rank", "hardware_id", "link",
                "physical_transfer_count", "action_counts",
                "bytes_by_action", "transferred_bytes", "busy_time_s",
                "idle_time_s", "utilization",
            },
            label,
        )
        pool_name = record["pool_name"]
        hardware_id = record["hardware_id"]
        link = record["link"]
        if (not isinstance(pool_name, str) or not pool_name
                or not isinstance(hardware_id, str) or not hardware_id
                or not isinstance(link, str) or not link):
            raise ValueError(f"{label} identity fields must be non-empty")
        rank = _nonnegative_integer(record["rank"], f"{label} rank")
        if resource_id != f"pool:{pool_name}:rank:{rank}:{link}":
            raise ValueError(f"{label} ID disagrees with pool/rank/link")
        physical_count = _nonnegative_integer(
            record["physical_transfer_count"],
            f"{label} physical transfer count",
        )
        if physical_count == 0:
            raise ValueError(f"{label} must contain a physical transfer")
        action_counts = _mapping(
            record["action_counts"], f"{label} action counts")
        bytes_by_action = _mapping(
            record["bytes_by_action"], f"{label} bytes by action")
        if (not action_counts
                or set(action_counts) != set(bytes_by_action)
                or not set(action_counts).issubset({"evict", "load"})):
            raise ValueError(f"{label} action coverage is invalid")
        normalized_counts = {
            action: _nonnegative_integer(
                action_counts[action], f"{label} {action} count")
            for action in action_counts
        }
        normalized_bytes = {
            action: _nonnegative_integer(
                bytes_by_action[action], f"{label} {action} bytes")
            for action in bytes_by_action
        }
        if (any(value == 0 for value in normalized_counts.values())
                or any(value == 0 for value in normalized_bytes.values())
                or sum(normalized_counts.values()) != physical_count):
            raise ValueError(f"{label} action totals are invalid")
        transferred_bytes = _nonnegative_integer(
            record["transferred_bytes"], f"{label} transferred bytes")
        if (transferred_bytes == 0
                or transferred_bytes != sum(normalized_bytes.values())):
            raise ValueError(f"{label} byte total is invalid")
        busy_time_s = _nonnegative_number(
            record["busy_time_s"], f"{label} busy time")
        idle_time_s = _nonnegative_number(
            record["idle_time_s"], f"{label} idle time")
        utilization = _nonnegative_number(
            record["utilization"], f"{label} utilization")
        if (not math.isclose(
                busy_time_s + idle_time_s,
                duration_s,
                rel_tol=1e-12,
                abs_tol=1e-15)
                or not math.isclose(
                    utilization,
                    busy_time_s / duration_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15)):
            raise ValueError(f"{label} utilization arithmetic is invalid")


def _validate_host_kv_resource_binding(
    artifact_value: object,
    resources_value: object,
    point_label: str,
) -> None:
    artifact = _mapping(
        artifact_value, f"{point_label} execution artifact")
    resources = _mapping(
        resources_value, f"{point_label} system resources")
    modeled = _mapping(
        artifact["modeled_kv_offload"],
        f"{point_label} modeled KV offload",
    )
    if (artifact["schema_version"] not in (7, 8)
            or modeled["schema_version"] != 4):
        if (resources["schema_version"] == 5
                and _mapping(
                    resources["host_kv_link_utilization"],
                    f"{point_label} Host-KV link utilization")):
            raise ValueError(
                f"{point_label} legacy/disabled execution artifact cannot "
                "claim Host-KV link utilization")
        return
    if resources["schema_version"] != 5:
        raise ValueError(
            f"{point_label} execution artifact v7 requires "
            "SystemResourceManifest v5")

    if modeled["schema_version"] != 4:
        raise ValueError(
            f"{point_label} execution artifact v7 requires modeled KV "
            "offload v4")
    steps_by_id = {}
    for raw_step in _sequence(
            artifact["steps"], f"{point_label} execution steps"):
        step = _mapping(raw_step, f"{point_label} execution step")
        step_id = step["step_id"]
        if step_id in steps_by_id:
            raise ValueError(
                f"{point_label} has duplicate execution step IDs")
        steps_by_id[step_id] = step

    records_by_resource = {}
    for raw_transfer in _sequence(
            modeled["transfers"], f"{point_label} modeled KV transfers"):
        transfer = _mapping(
            raw_transfer, f"{point_label} modeled KV transfer")
        step_id = transfer["step_id"]
        if step_id not in steps_by_id:
            raise ValueError(
                f"{point_label} Host-KV transfer references unknown step")
        step = steps_by_id[step_id]
        calls_by_id = {
            call["call_id"]: call
            for call in (
                _mapping(raw_call, f"{point_label} kernel call")
                for raw_call in _sequence(
                    _mapping(
                        step["kernel_plan"],
                        f"{point_label} kernel plan",
                    )["calls"],
                    f"{point_label} kernel calls",
                )
            )
        }
        action = _mapping(
            transfer["intent"], f"{point_label} KV intent")["action"]
        for raw_physical in _sequence(
                transfer["physical_transfers"],
                f"{point_label} physical KV transfers"):
            physical = _mapping(
                raw_physical, f"{point_label} physical KV transfer")
            resource_id = physical["resource_id"]
            call_id = physical["call_id"]
            if call_id not in calls_by_id:
                raise ValueError(
                    f"{point_label} Host-KV transfer lacks a kernel call")
            call = calls_by_id[call_id]
            completion = _mapping(
                physical["completion"],
                f"{point_label} physical KV completion",
            )
            interval = _mapping(
                physical["absolute_interval"],
                f"{point_label} physical KV absolute interval",
            )
            contract = (
                physical["pool_name"],
                physical["rank"],
                _mapping(
                    call["kernel_spec"],
                    f"{point_label} KV KernelSpec",
                )["hardware_id"],
                completion["link"],
            )
            if resource_id not in records_by_resource:
                records_by_resource[resource_id] = {
                    "contract": contract,
                    "intervals": [],
                    "physical_transfer_count": 0,
                    "action_counts": {},
                    "bytes_by_action": {},
                }
            record = records_by_resource[resource_id]
            if record["contract"] != contract:
                raise ValueError(
                    f"{point_label} Host-KV resource contract changed")
            record["intervals"].append((
                interval["start_time_s"], interval["end_time_s"]))
            record["physical_transfer_count"] += 1
            if action not in record["action_counts"]:
                record["action_counts"][action] = 0
                record["bytes_by_action"][action] = 0
            record["action_counts"][action] += 1
            record["bytes_by_action"][action] += completion["bytes"]

    window = _mapping(
        resources["observation_window"],
        f"{point_label} observation window",
    )
    duration_s = _positive_number(
        window["duration_s"], f"{point_label} observation duration")
    schedule_entries = _sequence(
        _mapping(
            artifact["schedule_trace"],
            f"{point_label} schedule trace",
        )["entries"],
        f"{point_label} schedule entries",
    )
    if not schedule_entries:
        raise ValueError(
            f"{point_label} execution artifact v7 has no schedule entries")
    schedule_completion_s = max(
        _nonnegative_number(
            _mapping(
                raw_entry,
                f"{point_label} schedule entry {index}",
            )["end_time_s"],
            f"{point_label} schedule entry {index} end time",
        )
        for index, raw_entry in enumerate(schedule_entries)
    )
    if not math.isclose(
            duration_s,
            schedule_completion_s,
            rel_tol=1e-12,
            abs_tol=1e-15):
        raise ValueError(
            f"{point_label} Host-KV observation window disagrees with "
            "schedule completion")
    expected = {}
    for resource_id in sorted(records_by_resource):
        record = records_by_resource[resource_id]
        intervals = sorted(record["intervals"])
        previous_end_s = None
        busy_time_s = 0.0
        for start_time_s, end_time_s in intervals:
            if (start_time_s < 0.0
                    or end_time_s <= start_time_s
                    or end_time_s > duration_s + 1e-15):
                raise ValueError(
                    f"{point_label} Host-KV interval lies outside the "
                    "observation window")
            if (previous_end_s is not None
                    and start_time_s < previous_end_s - 1e-15):
                raise ValueError(
                    f"{point_label} Host-KV intervals overlap on "
                    f"{resource_id!r}")
            busy_time_s += end_time_s - start_time_s
            previous_end_s = end_time_s
        idle_time_s = duration_s - busy_time_s
        if idle_time_s < -1e-15:
            raise ValueError(
                f"{point_label} Host-KV busy time exceeds observation window")
        idle_time_s = max(0.0, idle_time_s)
        pool_name, rank, hardware_id, link = record["contract"]
        bytes_by_action = {
            action: record["bytes_by_action"][action]
            for action in sorted(record["bytes_by_action"])
        }
        expected[resource_id] = {
            "pool_name": pool_name,
            "rank": rank,
            "hardware_id": hardware_id,
            "link": link,
            "physical_transfer_count": record["physical_transfer_count"],
            "action_counts": {
                action: record["action_counts"][action]
                for action in sorted(record["action_counts"])
            },
            "bytes_by_action": bytes_by_action,
            "transferred_bytes": sum(bytes_by_action.values()),
            "busy_time_s": busy_time_s,
            "idle_time_s": idle_time_s,
            "utilization": busy_time_s / duration_s,
        }
    actual = _mapping(
        resources["host_kv_link_utilization"],
        f"{point_label} Host-KV link utilization",
    )
    if actual != expected or not expected:
        raise ValueError(
            f"{point_label} Host-KV link utilization disagrees with "
            "execution artifact v7")


def _validate_critical_path(value: object, point_label: str) -> None:
    path = _mapping(value, f"{point_label} critical path")
    _exact_fields(
        path,
        {"schema_version", "semantics", "steps", "requests", "system_completion"},
        f"{point_label} critical path",
    )
    if path["schema_version"] != 2:
        raise ValueError(f"{point_label} critical path schema_version must be 2")
    _mapping(path["semantics"], f"{point_label} critical-path semantics")
    steps = _sequence(path["steps"], f"{point_label} critical-path steps")
    for index, step_value in enumerate(steps):
        step = _mapping(step_value, f"{point_label} critical-path step {index}")
        _required_fields(
            step,
            {
                "step_id",
                "pool_name",
                "duration_s",
                "canonical_path_unique",
                "contribution_s_by_category",
                "dominant_category",
            },
            f"{point_label} critical-path step {index}",
        )
        _nonnegative_number(
            step["duration_s"],
            f"{point_label} critical-path step {index} duration",
        )
        if not isinstance(step["canonical_path_unique"], bool):
            raise ValueError(
                f"{point_label} critical-path step {index} unique flag "
                "must be boolean")
        _validate_contributions(
            step["contribution_s_by_category"],
            f"{point_label} critical-path step {index}",
        )
    requests = _mapping(
        path["requests"], f"{point_label} critical-path requests")
    for request_id, request_value in requests.items():
        request = _mapping(
            request_value, f"{point_label} critical-path request {request_id}")
        _exact_fields(
            request, {"arrival_time_s", "attribution", "ttft", "e2e"},
            f"{point_label} critical-path request {request_id}")
        _nonnegative_number(
            request["arrival_time_s"],
            f"{point_label} critical-path request {request_id} arrival",
        )
        if not isinstance(request["attribution"], str):
            raise ValueError(
                f"{point_label} critical-path request {request_id} "
                "attribution must be a string")
        for target_name in ("ttft", "e2e"):
            target = _mapping(
                request[target_name],
                f"{point_label} request {request_id} {target_name}",
            )
            _required_fields(
                target, {"availability", "duration_s", "dominant_category"},
                f"{point_label} request {request_id} {target_name}",
            )
            if target["availability"] == "available":
                _nonnegative_number(
                    target["duration_s"],
                    f"{point_label} request {request_id} {target_name} duration",
                )
            elif (target["availability"]
                  == "not_applicable_zero_output_tokens"):
                if (target["duration_s"] is not None
                        or target["dominant_category"] is not None):
                    raise ValueError(
                        f"{point_label} unavailable request path has values")
            else:
                raise ValueError(
                    f"{point_label} request path availability is unsupported")
    completion = _mapping(
        path["system_completion"], f"{point_label} system completion")
    _required_fields(
        completion,
        {"time_s", "terminal_request_ids", "terminal_observation_paths"},
        f"{point_label} system completion",
    )
    _positive_number(completion["time_s"], f"{point_label} system completion time")
    terminal_paths = _sequence(
        completion["terminal_observation_paths"],
        f"{point_label} terminal observation paths",
    )
    if not terminal_paths:
        raise ValueError(f"{point_label} has no terminal critical path")
    for index, terminal_value in enumerate(terminal_paths):
        terminal = _mapping(
            terminal_value, f"{point_label} terminal path {index}")
        _exact_fields(
            terminal,
            {
                "request_id",
                "path_start_time_s",
                "path_duration_s",
                "unattributed_pre_arrival_window_s",
                "contribution_s_by_category",
                "dominant_category",
            },
            f"{point_label} terminal path {index}",
        )
        if not isinstance(terminal["request_id"], str):
            raise ValueError(
                f"{point_label} terminal path {index} request ID is invalid")
        for field in (
                "path_start_time_s",
                "path_duration_s",
                "unattributed_pre_arrival_window_s"):
            _nonnegative_number(
                terminal[field], f"{point_label} terminal path {index} {field}")
        _validate_contributions(
            terminal["contribution_s_by_category"],
            f"{point_label} terminal path {index}",
        )


def _validate_contributions(value: object, label: str) -> None:
    contributions = _mapping(value, f"{label} contributions")
    for category, duration in contributions.items():
        if not isinstance(category, str) or not category:
            raise ValueError(f"{label} contribution category is invalid")
        _nonnegative_number(duration, f"{label} contribution {category}")


def _validate_sweep_analysis(
    sweep: Mapping[str, object],
    points: Sequence[Mapping[str, object]],
) -> None:
    tradeoff = _mapping(sweep["interactivity_tgs"], "Interactivity-TGS analysis")
    peak = _mapping(sweep["peak_selection"], "sampled peak selection")
    saturation = _mapping(sweep["saturation_analysis"], "saturation analysis")
    _required_fields(
        tradeoff,
        {"availability", "contract", "points", "pareto_frontier"},
        "Interactivity-TGS analysis",
    )
    _required_fields(
        peak, {"availability", "contract"}, "sampled peak selection")
    _required_fields(
        saturation,
        {"availability", "contract", "intervals"},
        "saturation analysis",
    )
    if tradeoff["contract"] is None:
        expected = arrival_rate_sweep_analysis_manifest(None, None)
    else:
        contract = _interactivity_tgs_contract(tradeoff["contract"])
        canonical_points = tuple(
            sweep_metric_point_from_manifest(
                point["arrival_rate_scale"],
                _mapping(point["result"], "sweep point result")["metrics"],
                contract,
            )
            for point in points
        )
        expected = arrival_rate_sweep_analysis_manifest(
            canonical_points, contract)
    for analysis_name, actual in (
            ("interactivity_tgs", tradeoff),
            ("peak_selection", peak),
            ("saturation_analysis", saturation)):
        if (_canonical_json_bytes(actual)
                != _canonical_json_bytes(expected[analysis_name])):
            raise ValueError(
                f"{analysis_name} disagrees with canonical sweep metrics")


def _interactivity_tgs_contract(value: object) -> InteractivityTGSContract:
    contract = _mapping(value, "Interactivity-TGS contract")
    expected_fields = {
        "schema_version",
        "contract_id",
        "interactivity_metric",
        "interactivity_percentile",
        "interactivity_unit",
        "interactivity_aggregation",
        "interactivity_eligibility",
        "tgs_metric",
        "tgs_unit",
        "tgs_aggregation",
        "peak_selection",
        "saturation_rule",
        "saturation_marginal_gain_ratio_threshold",
        "saturation_minimum_consecutive_intervals",
        "source_path",
        "source_sha256",
    }
    _exact_fields(contract, expected_fields, "Interactivity-TGS contract")
    return InteractivityTGSContract(
        schema_version=contract["schema_version"],
        contract_id=contract["contract_id"],
        interactivity_metric=contract["interactivity_metric"],
        interactivity_percentile=contract["interactivity_percentile"],
        interactivity_unit=contract["interactivity_unit"],
        interactivity_aggregation=contract["interactivity_aggregation"],
        interactivity_eligibility=contract["interactivity_eligibility"],
        tgs_metric=contract["tgs_metric"],
        tgs_unit=contract["tgs_unit"],
        tgs_aggregation=contract["tgs_aggregation"],
        peak_selection=contract["peak_selection"],
        saturation_rule=contract["saturation_rule"],
        saturation_marginal_gain_ratio_threshold=contract[
            "saturation_marginal_gain_ratio_threshold"],
        saturation_minimum_consecutive_intervals=contract[
            "saturation_minimum_consecutive_intervals"],
        source_path=contract["source_path"],
        source_sha256=contract["source_sha256"],
    )


def _render_execution_provenance(provenance: Mapping[str, object]) -> list[str]:
    repository = _mapping(provenance["repository"], "repository provenance")
    hardware_ids = _sequence(provenance["loaded_hardware_ids"], "loaded hardware IDs")
    configs = _mapping(provenance["hardware_configs"], "hardware configs")
    lines = [
        f"- Git commit: `{repository['git_commit']}`",
        f"- Worktree state: {_code(repository['worktree_state'])}",
        f"- Worktree diff hash: `{repository['git_diff_hash']}`",
        "- Loaded hardware IDs: "
        + ", ".join(_code(value) for value in hardware_ids),
        f"- Prediction method: {_code(provenance['prediction_method'])}",
        "- Hardware config identities:",
    ]
    for role in sorted(configs):
        config = configs[role]
        if config is None:
            lines.append(f"  - {_code(role)}: not applicable")
        elif isinstance(config, Mapping):
            if "sha256" not in config or "path" not in config:
                raise ValueError(
                    f"hardware config provenance {role!r} lacks path or SHA256")
            _sha256(config["sha256"], f"hardware config {role} SHA256")
            lines.append(
                f"  - {_code(role)}: "
                f"{_code(config['path'])}; SHA256 "
                f"`{config['sha256']}`")
        elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            if not config:
                lines.append(f"  - {_code(role)}: none")
            for index, item in enumerate(config):
                additional = _mapping(
                    item, f"additional hardware config {index}")
                _required_fields(
                    additional, {"path", "sha256"},
                    f"additional hardware config {index}")
                _sha256(
                    additional["sha256"],
                    f"additional hardware config {index} SHA256",
                )
                lines.append(
                    f"  - {_code(f'{role}[{index}]')}: "
                    f"{_code(additional['path'])}; SHA256 "
                    f"`{additional['sha256']}`")
        else:
            raise ValueError(f"hardware config provenance {role!r} is invalid")
    launch_inputs = _mapping(provenance["launch_inputs"], "launch inputs")
    launch_data = _mapping(launch_inputs["launch_data"], "launch data")
    lines.extend([
        f"- Launch policy: {_code(launch_inputs['launch_policy'])}",
        "- Launch-data identities:",
    ])
    for hardware_id in sorted(launch_data):
        launch = _mapping(launch_data[hardware_id], "launch-data identity")
        _required_fields(
            launch, {"path", "tree_sha256", "file_count"},
            f"launch data {hardware_id}")
        _sha256(launch["tree_sha256"], f"launch data {hardware_id} tree SHA256")
        lines.append(
            f"  - {_code(hardware_id)}: "
            f"{_code(launch['path'])}; tree SHA256 "
            f"`{launch['tree_sha256']}`; files `{launch['file_count']}`")
    lines.extend([
        "- Runtime inputs: " + _inline_json(provenance["runtime_inputs"]),
        "- Microbenchmark inputs: "
        + _inline_json(provenance["microbenchmark_inputs"]),
        "- External predictor inputs: "
        + _inline_json(provenance["external_predictor_inputs"]),
        "",
    ])
    return lines


def _render_sweep_analysis(sweep: Mapping[str, object]) -> list[str]:
    tradeoff = _mapping(sweep["interactivity_tgs"], "Interactivity-TGS analysis")
    peak = _mapping(sweep["peak_selection"], "peak selection")
    saturation = _mapping(sweep["saturation_analysis"], "saturation analysis")
    lines = ["## Interactivity–TGS analysis", ""]
    if tradeoff["availability"] == "unavailable_no_metric_contract":
        lines.extend([
            "Unavailable: no metric contract was supplied; no "
            "Interactivity–TGS metric default applied.",
            "",
            "Peak and saturation are also unavailable because a sampled "
            "trade-off metric was not defined.",
            "",
        ])
        return lines

    contract = _mapping(tradeoff["contract"], "Interactivity-TGS contract")
    lines.extend([
        f"- Availability: `{tradeoff['availability']}`",
        f"- Contract: {_code(contract['contract_id'])}",
        f"- Contract source SHA256: `{contract['source_sha256']}`",
        "- Definition: "
        f"P{contract['interactivity_percentile']} "
        f"{_code(contract['interactivity_metric'])} versus "
        f"{_code(contract['tgs_metric'])}.",
        "",
        "| Scale | Interactivity (s) | TGS | Request throughput/s |",
        "|---:|---:|---:|---:|",
    ])
    for point in _sequence(tradeoff["points"], "Interactivity-TGS points"):
        row = _mapping(point, "Interactivity-TGS point")
        _required_fields(
            row,
            {"arrival_rate_scale", "interactivity_value_s", "tgs_value",
             "request_throughput_per_s"},
            "Interactivity-TGS point",
        )
        lines.append(
            f"| {_format(row['arrival_rate_scale'])} "
            f"| {_format(row['interactivity_value_s'])} "
            f"| {_format(row['tgs_value'])} "
            f"| {_format(row['request_throughput_per_s'])} |")
    lines.extend(["", "Pareto frontier:", ""])
    frontier = _sequence(tradeoff["pareto_frontier"], "Interactivity-TGS frontier")
    if frontier:
        for point in frontier:
            row = _mapping(point, "Interactivity-TGS frontier point")
            lines.append(
                f"- scale `{_format(row['arrival_rate_scale'])}`: "
                f"interactivity `{_format(row['interactivity_value_s'])}` s, "
                f"TGS `{_format(row['tgs_value'])}`")
    else:
        lines.append("- No eligible Pareto point.")
    lines.extend(["", "### Observed sampled peak", ""])
    lines.append(f"- Availability: `{peak['availability']}`")
    if "selection_scope" in peak:
        lines.append(f"- Scope: {_code(peak['selection_scope'])}")
    if "point" in peak and peak["point"] is not None:
        peak_point = _mapping(peak["point"], "sampled peak point")
        lines.append(
            f"- Point: scale `{_format(peak_point['arrival_rate_scale'])}`, "
            f"TGS `{_format(peak_point['tgs_value'])}`.")
    lines.extend(["", "### Saturation", ""])
    lines.extend([
        f"- Availability: `{saturation['availability']}`",
        "- Scope: sampled arrival-rate scales only.",
        "",
        "| From scale | To scale | Marginal TGS gain ratio | Classification |",
        "|---:|---:|---:|---|",
    ])
    for interval in _sequence(saturation["intervals"], "saturation intervals"):
        row = _mapping(interval, "saturation interval")
        lines.append(
            f"| {_format(row['from_arrival_rate_scale'])} "
            f"| {_format(row['to_arrival_rate_scale'])} "
            f"| {_format(row['marginal_tgs_gain_ratio'])} "
            f"| {_table_code(row['classification'])} |")
    lines.append("")
    return lines


def _render_point(index: int, point: Mapping[str, object]) -> list[str]:
    result = _mapping(point["result"], "point result")
    metrics = _mapping(result["metrics"], "point result metrics")
    artifact = _mapping(result["execution_artifact"], "execution artifact")
    lines = [
        f"## Sweep point {index}: arrival-rate scale "
        f"{_format(point['arrival_rate_scale'])}",
        "",
        f"- Request-stream SHA256: `{point['request_stream_sha256']}`",
        f"- Execution artifact SHA256: `{artifact['artifact_sha256']}`",
        *_render_timing_scope(result["performance_timing_status"]),
        "",
        "### Throughput",
        "",
        "| Metric | Value per second |",
        "|---|---:|",
        "| Requests | "
        f"{_format(metrics['request_throughput_per_s'])} |",
        "| Output tokens | "
        f"{_format(metrics['output_token_throughput_per_s'])} |",
        "| Input + output tokens | "
        f"{_format(metrics['token_throughput_per_s'])} |",
        "",
    ]
    lines.extend(_render_latency_distributions(metrics["latency_distributions"]))
    lines.extend(_render_slo(metrics["slo_goodput"]))
    lines.extend(_render_resources(metrics["system_resources"]))
    lines.extend(_render_critical_path(metrics["critical_path"]))
    return lines


def _render_latency_distributions(value: object) -> list[str]:
    distributions = _mapping(value, "latency distributions")
    lines = ["### Latency CDF and P0–P100", ""]
    labels = {
        "ttft_s": "TTFT",
        "tpot_s": "TPOT",
        "latency_s": "E2E latency",
    }
    for metric_name in ("ttft_s", "tpot_s", "latency_s"):
        distribution = _mapping(distributions[metric_name], metric_name)
        lines.extend([
            f"#### {labels[metric_name]}",
            "",
            f"- Availability: `{distribution['availability']}`",
            f"- Eligible requests: {distribution['eligible_request_count']}",
            f"- Excluded requests: {distribution['excluded_request_count']} "
            f"({_code(distribution['exclusion_reason'])})",
            "",
        ])
        if distribution["availability"] != "available":
            continue
        quantiles = _mapping(distribution["quantiles"], "latency quantiles")
        percentiles = _sequence(quantiles["percentiles"], "latency percentiles")
        values = _sequence(quantiles["values_s"], "latency quantile values")
        lines.extend([
            "Quantiles (seconds):",
            "",
            ", ".join(
                f"P{percentile}={_format(metric_value)}"
                for percentile, metric_value in zip(percentiles, values)
            ),
            "",
            "Empirical CDF:",
            "",
            "| Latency (s) | Cumulative probability |",
            "|---:|---:|",
        ])
        cdf = _mapping(distribution["empirical_cdf"], "latency CDF")
        for latency, probability in zip(
                _sequence(cdf["values_s"], "latency CDF values"),
                _sequence(cdf["cumulative_probabilities"],
                          "latency CDF probabilities")):
            lines.append(f"| {_format(latency)} | {_format(probability)} |")
        lines.append("")
    return lines


def _render_slo(value: object) -> list[str]:
    slo = _mapping(value, "SLO goodput")
    lines = ["### SLO goodput", ""]
    if slo["availability"] == "unavailable_no_slo_contract":
        lines.extend([
            "Unavailable: no SLO contract was supplied; no SLO threshold "
            "default applied.",
            "",
        ])
        return lines
    contract = _mapping(slo["contract"], "SLO contract")
    lines.extend([
        f"- Contract: {_code(contract['contract_id'])}",
        f"- Contract source SHA256: `{contract['source_sha256']}`",
        f"- TTFT threshold: `{_format(contract['ttft_threshold_s'])}` s",
        f"- TPOT threshold: `{_format(contract['tpot_threshold_s'])}` s",
        f"- Request SLO goodput: `{_format(slo['request_goodput_per_s'])}` /s",
        f"- Token SLO goodput: `{_format(slo['token_goodput_per_s'])}` /s",
        "- Population: " + _inline_json(slo["population"]),
        "- Violations: " + _inline_json(slo["violation_counts"]),
        "",
    ])
    return lines


def _render_resources(value: object) -> list[str]:
    resources = _mapping(value, "system resources")
    window = _mapping(resources["observation_window"], "observation window")
    lines = [
        "### Resource utilization and queues",
        "",
        f"Observation window: `{_format(window['duration_s'])}` s "
        f"({_code(window['basis'])}).",
        "",
        "Worker pools:",
        "",
        "| Pool | Kernel intervals | Busy (s) | Idle (s) | Utilization |",
        "|---|---:|---:|---:|---:|",
    ]
    worker_pools = _mapping(resources["worker_pool_utilization"], "worker pools")
    for pool_name in sorted(worker_pools):
        record = _mapping(worker_pools[pool_name], f"worker pool {pool_name}")
        lines.append(
            f"| {_table_code(pool_name)} "
            f"| {_format(record['kernel_interval_count'])} "
            f"| {_format(record['busy_time_s'])} "
            f"| {_format(record['idle_time_s'])} "
            f"| {_format(record['utilization'])} |")
    lines.extend(["", "Transfer links:", ""])
    links = _mapping(resources["link_utilization"], "link utilization")
    if not links:
        lines.append("- None.")
    for resource_id in sorted(links):
        lines.append(
            f"- {_code(resource_id)}: "
            + _inline_json(links[resource_id]))
    lines.extend(["", "Communication resources:", ""])
    communication = _mapping(
        resources["communication_resource_utilization"],
        "communication resource utilization",
    )
    if not communication:
        lines.append("- None.")
    for resource_id in sorted(communication):
        lines.append(
            f"- {_code(resource_id)}: "
            + _inline_json(communication[resource_id]))
    if resources["schema_version"] == 5:
        lines.extend(["", "Host-KV rank-local links:", ""])
        host_kv_links = _mapping(
            resources["host_kv_link_utilization"],
            "Host-KV link utilization",
        )
        if not host_kv_links:
            lines.append("- None.")
        for resource_id in sorted(host_kv_links):
            lines.append(
                f"- {_code(resource_id)}: "
                + _inline_json(host_kv_links[resource_id]))
    router = _mapping(resources["router_queue"], "router queue")
    scheduler = _mapping(resources["scheduler_queue"], "scheduler queue")
    lines.extend([
        "",
        "Router queue:",
        "",
        f"- Availability: {_code(router['availability'])}",
        f"- Overall: {_inline_json(router['overall'])}",
        f"- By stage: {_inline_json(router['by_stage'])}",
    ])
    if "dispatch_service" in router:
        lines.append(
            f"- Dispatch service: {_inline_json(router['dispatch_service'])}")
    lines.extend([
        "",
        "Scheduler queue:",
        "",
        f"- Availability: {_code(scheduler['availability'])}",
        f"- Overall: {_inline_json(scheduler['overall'])}",
        f"- By stage: {_inline_json(scheduler['by_stage'])}",
        f"- By pool: {_inline_json(scheduler['by_pool'])}",
        "",
    ])
    return lines


def _render_critical_path(value: object) -> list[str]:
    path = _mapping(value, "critical path")
    semantics = _mapping(path["semantics"], "critical-path semantics")
    completion = _mapping(path["system_completion"], "system completion")
    lines = ["### Modeled critical path", "", "Semantics:", ""]
    for name in sorted(semantics):
        lines.append(
            f"- {_code(name)}: {_code(semantics[name])}")
    lines.extend([
        "",
        "Step paths:",
        "",
        "| Step | Pool | Duration (s) | Unique | Dominant category | Contributions |",
        "|---|---|---:|---|---|---|",
    ])
    for step_value in _sequence(path["steps"], "critical-path steps"):
        step = _mapping(step_value, "critical-path step")
        lines.append(
            f"| {_table_code(step['step_id'])} "
            f"| {_table_code(step['pool_name'])} "
            f"| {_format(step['duration_s'])} "
            f"| {_format(step['canonical_path_unique'])} "
            f"| {_table_code(step['dominant_category'])} "
            f"| {_markdown_cell(_inline_json(step['contribution_s_by_category']))} |")
    lines.extend([
        "",
        "Request paths:",
        "",
        "| Request | Attribution | TTFT duration (s) | TTFT dominant "
        "| E2E duration (s) | E2E dominant |",
        "|---|---|---:|---|---:|---|",
    ])
    requests = _mapping(path["requests"], "critical-path requests")
    for request_id in sorted(requests):
        request = _mapping(requests[request_id], "critical-path request")
        ttft = _mapping(request["ttft"], "request TTFT path")
        e2e = _mapping(request["e2e"], "request E2E path")
        lines.append(
            f"| {_table_code(request_id)} "
            f"| {_table_code(request['attribution'])} "
            f"| {_format(ttft['duration_s'])} "
            f"| {_table_code(ttft['dominant_category'])} "
            f"| {_format(e2e['duration_s'])} "
            f"| {_table_code(e2e['dominant_category'])} |")
    lines.extend([
        "",
        f"System completion: `{_format(completion['time_s'])}` s.",
        "",
        "| Terminal request | Path start (s) | Duration (s) "
        "| Dominant category | Contributions |",
        "|---|---:|---:|---|---|",
    ])
    terminal_paths = _sequence(
        completion["terminal_observation_paths"], "terminal observation paths")
    for terminal_path in terminal_paths:
        row = _mapping(terminal_path, "terminal observation path")
        lines.append(
            f"| {_table_code(row['request_id'])} "
            f"| {_format(row['path_start_time_s'])} "
            f"| {_format(row['path_duration_s'])} "
            f"| {_table_code(row['dominant_category'])} "
            f"| {_markdown_cell(_inline_json(row['contribution_s_by_category']))} |")
    lines.append("")
    return lines




def _sequence(value: object, label: str) -> Sequence[object]:
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))):
        raise TypeError(f"{label} must be a sequence")
    return value


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    supplied = set(value)
    if supplied != expected:
        raise ValueError(
            f"{label} fields disagree with schema: "
            f"missing={sorted(expected - supplied)}, "
            f"unexpected={sorted(supplied - expected)}")


def _required_fields(
    value: Mapping[str, object],
    required: set[str],
    label: str,
) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")


def _sha256(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be lowercase SHA256 hex")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_number(value: object, label: str) -> float:
    normalized = _nonnegative_number(value, label)
    if normalized <= 0:
        raise ValueError(f"{label} must be positive")
    return normalized


def _nonnegative_number(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nondecreasing_nonnegative(values: Sequence[object], label: str) -> None:
    normalized = tuple(_nonnegative_number(value, label) for value in values)
    if tuple(sorted(normalized)) != normalized:
        raise ValueError(f"{label} must be nondecreasing")


def _format(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError("report cannot format a non-finite number")
        return format(value, ".12g")
    return str(value)


def _inline_json(value: object) -> str:
    try:
        return _code(json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ))
    except (TypeError, ValueError) as error:
        raise ValueError("report field is not canonical JSON") from error


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("report field is not canonical JSON") from error


def _code(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    backtick_run_lengths = tuple(
        len(match.group(0)) for match in re.finditer(r"`+", text)
    )
    longest_backtick_run = (
        max(backtick_run_lengths) if backtick_run_lengths else 0)
    delimiter = "`" * (longest_backtick_run + 1)
    needs_padding = (
        text.startswith(("`", " ")) or text.endswith(("`", " ")))
    padding = " " if needs_padding else ""
    return f"{delimiter}{padding}{text}{padding}{delimiter}"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table_code(value: object) -> str:
    if value is None:
        return "N/A"
    return _markdown_cell(_code(value))
