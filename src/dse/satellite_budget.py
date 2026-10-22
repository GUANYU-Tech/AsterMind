"""Satellite budget analysis helpers for fair PD and DSE comparisons."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..orbit import OrbitTraceBundle as RuntimeTraceBundle, OrbitTraceManifest as RuntimeTraceManifest, load_orbit_trace as load_surrogate_trace_bundle


def analyze_communication_window_satellite_budget(
    *,
    simulation_result: Any | None = None,
    trace_bundle: RuntimeTraceBundle | None = None,
    workload_generation: Any | None = None,
    serving_throughput: Any | None = None,
    config: dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze visible/access-window-derived satellite budget candidates."""

    analysis_config = dict(config or {})
    if not bool(analysis_config.get("enabled", False)):
        return {
            "evaluated": False,
            "reason": "communication_window_budget disabled",
            "warnings": (),
        }
    warnings_out: list[str] = []
    bundle = trace_bundle
    if bundle is None:
        manifest = RuntimeTraceManifest.from_value(getattr(simulation_result, "trace_manifest", None))
        if not manifest.evaluated:
            return {
                "evaluated": False,
                "reason": "simulation trace manifest is not evaluated",
                "warnings": tuple(dict.fromkeys((*manifest.warnings,))),
            }
        try:
            bundle = load_surrogate_trace_bundle(manifest, base_dir=base_dir)
        except Exception as exc:  # pragma: no cover - exact loader failures vary by caller.
            return {
                "evaluated": False,
                "reason": f"failed to load surrogate trace bundle: {exc}",
                "warnings": (f"communication_window_budget analysis unavailable: {exc}",),
            }
    warnings_out.extend(bundle.warnings)
    return analyze_communication_window_satellite_budget_from_bundle(
        bundle,
        workload_generation=workload_generation,
        serving_throughput=serving_throughput,
        config=analysis_config,
        warnings=tuple(warnings_out),
    )


def analyze_communication_window_satellite_budget_from_bundle(
    trace_bundle: RuntimeTraceBundle,
    *,
    workload_generation: Any | None = None,
    serving_throughput: Any | None = None,
    config: dict[str, Any] | None = None,
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    analysis_config = dict(config or {})
    warnings_out = list(warnings)
    visible_counts = _visible_counts(_visible_count_samples(trace_bundle))
    visible_satellite_ids = _visible_satellite_ids(trace_bundle)
    request_observations = _request_visible_counts(trace_bundle, workload_generation)
    if not visible_counts and not request_observations:
        return {
            "evaluated": False,
            "reason": "trace has no visible_count_samples or request/access-window observations",
            "analysis_scope": "communication_window_static_surrogate",
            "valid_for_paper_main_results": False,
            "warnings": tuple(dict.fromkeys((*warnings_out, "communication_window_budget analysis needs trace access windows"))),
        }

    counts_for_candidates = [count for count in (request_observations or visible_counts) if count >= 0]
    candidate_budgets = _candidate_budgets(
        analysis_config,
        visible_counts=visible_counts,
        request_counts=request_observations,
        visible_satellite_count=len(visible_satellite_ids),
        serving_throughput=serving_throughput,
    )
    if not candidate_budgets:
        candidate_budgets = (1,)
        warnings_out.append("communication_window_budget had no positive candidates; using satellite_budget_used=1")

    rows = tuple(
        _candidate_row(
            budget,
            counts_for_candidates=counts_for_candidates,
            constellation_count=_constellation_count(trace_bundle, visible_satellite_ids),
            access_windows=trace_bundle.sgl_windows,
        )
        for budget in candidate_budgets
    )
    selected = _select_budget(rows, target_coverage_ratio=float(analysis_config.get("target_coverage_ratio", 0.95)))
    best_performance = _best_performance_budget(rows)
    summary = _visible_count_summary(
        visible_counts=visible_counts,
        request_counts=request_observations,
        access_windows=trace_bundle.sgl_windows,
    )
    max_ids = int(analysis_config.get("max_visible_satellite_ids", 64))
    return {
        "evaluated": True,
        "analysis_scope": "communication_window_static_surrogate",
        "trace_model": trace_bundle.trace_model,
        "trace_id": trace_bundle.trace_id,
        "valid_for_paper_main_results": False,
        "candidate_source": "derived_from_access_windows_and_visible_count_samples",
        "visible_count_summary": summary,
        "visible_satellite_count": len(visible_satellite_ids),
        "visible_satellite_ids": visible_satellite_ids[:max(0, max_ids)],
        "visible_satellite_ids_truncated": len(visible_satellite_ids) > max(0, max_ids),
        "candidate_budget_rows": rows,
        "selected_budget": selected,
        "best_performance_budget": best_performance,
        "cost_model": "allocated_compute_payload_fraction_proxy",
        "warnings": tuple(dict.fromkeys(warnings_out)),
    }


def _candidate_row(
    budget: int,
    *,
    counts_for_candidates: list[int],
    constellation_count: int,
    access_windows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    positive_counts = [count for count in counts_for_candidates if count > 0]
    if positive_counts:
        capacity_factor = sum(min(budget, count) / count for count in positive_counts) / len(positive_counts)
        access_coverage = sum(1 for count in counts_for_candidates if count > 0 and budget > 0) / len(counts_for_candidates)
    else:
        capacity_factor = 0.0
        access_coverage = 0.0
    cost_fraction = budget / max(1, constellation_count)
    cost_per_capacity = cost_fraction / max(capacity_factor, 1e-12)
    delay_summary = _access_delay_summary(access_windows)
    return {
        "budget_strategy_id": f"active_satellites_{budget}",
        "satellite_budget_used": budget,
        "budget_source": "candidate_satellite_budgets",
        "request_access_coverage_ratio": access_coverage,
        "window_limited_capacity_factor": capacity_factor,
        "served_ratio_proxy": capacity_factor,
        "allocated_satellite_fraction_proxy": cost_fraction,
        "cost_per_served_request_proxy": cost_per_capacity,
        "slo_cost_score": capacity_factor / max(cost_fraction, 1e-12),
        "access_propagation_delay_p95_s": delay_summary["p95_s"],
        "access_setup_time_p95_s": delay_summary["setup_p95_s"],
        "bottleneck": "communication_window_visible_satellite_budget" if capacity_factor < 1.0 else "not_window_budget_limited",
        "warnings": (),
    }


def _select_budget(rows: tuple[dict[str, Any], ...], *, target_coverage_ratio: float) -> dict[str, Any]:
    if not rows:
        return {}
    max_factor = max(float(row["window_limited_capacity_factor"]) for row in rows)
    threshold = max_factor * max(0.0, min(1.0, target_coverage_ratio))
    candidates = [row for row in rows if float(row["window_limited_capacity_factor"]) >= threshold]
    selected = min(candidates or rows, key=lambda row: (int(row["satellite_budget_used"]), float(row["cost_per_served_request_proxy"])))
    return {
        "selection_policy": "min_budget_reaching_target_fraction_of_best_window_capacity",
        "target_coverage_ratio": target_coverage_ratio,
        "satellite_budget_used": selected["satellite_budget_used"],
        "window_limited_capacity_factor": selected["window_limited_capacity_factor"],
        "cost_per_served_request_proxy": selected["cost_per_served_request_proxy"],
        "budget_strategy_id": selected["budget_strategy_id"],
    }


def _best_performance_budget(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    if not rows:
        return {}
    selected = max(rows, key=lambda row: (float(row["window_limited_capacity_factor"]), -int(row["satellite_budget_used"])))
    return {
        "selection_policy": "max_window_limited_capacity_factor",
        "satellite_budget_used": selected["satellite_budget_used"],
        "window_limited_capacity_factor": selected["window_limited_capacity_factor"],
        "budget_strategy_id": selected["budget_strategy_id"],
    }


def _candidate_budgets(
    config: dict[str, Any],
    *,
    visible_counts: list[int],
    request_counts: list[int],
    visible_satellite_count: int,
    serving_throughput: Any | None,
) -> tuple[int, ...]:
    raw = (
        config.get("candidate_satellite_budgets")
        or config.get("candidate_satellite_counts")
        or config.get("candidate_counts")
        or ("visible_p50", "visible_p95", "visible_max", "visible_unique")
    )
    candidates: list[int] = []
    for item in raw:
        candidate = _candidate_budget_value(
            item,
            visible_counts=visible_counts,
            request_counts=request_counts,
            visible_satellite_count=visible_satellite_count,
            serving_throughput=serving_throughput,
        )
        if candidate is not None and candidate > 0:
            candidates.append(candidate)
    return tuple(sorted(set(candidates)))


def _candidate_budget_value(
    value: Any,
    *,
    visible_counts: list[int],
    request_counts: list[int],
    visible_satellite_count: int,
    serving_throughput: Any | None,
) -> int | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return int(value)
    label = str(value).lower()
    if label in {"visible_unique", "unique_visible", "visible_satellite_count"}:
        return visible_satellite_count
    if label in {"visible_max", "max_visible"}:
        return max(visible_counts, default=visible_satellite_count)
    if label in {"visible_p50", "p50_visible", "median_visible"}:
        return math.ceil(_quantile(visible_counts, 0.50))
    if label in {"visible_p95", "p95_visible"}:
        return math.ceil(_quantile(visible_counts, 0.95))
    if label in {"request_visible_max", "request_weighted_visible_max"}:
        return max(request_counts, default=max(visible_counts, default=visible_satellite_count))
    if label in {"request_visible_p50", "request_weighted_visible_p50"}:
        return math.ceil(_quantile(request_counts or visible_counts, 0.50))
    if label in {"request_visible_p95", "request_weighted_visible_p95"}:
        return math.ceil(_quantile(request_counts or visible_counts, 0.95))
    if label in {"serving_budget", "current_serving_budget"}:
        return _serving_budget_count(serving_throughput)
    return None


def _visible_counts(samples: tuple[dict[str, Any], ...]) -> list[int]:
    counts: list[int] = []
    for sample in samples:
        value = _first_present(
            sample.get("externally_available_count"),
            sample.get("geometry_visible_count"),
            sample.get("visible_count"),
        )
        if value is not None:
            counts.append(max(0, int(value)))
    return counts


def _visible_satellite_ids(trace_bundle: RuntimeTraceBundle) -> tuple[str, ...]:
    ids: set[str] = set()
    for window in trace_bundle.sgl_windows:
        satellite_id = window.get("satellite_id")
        if satellite_id is not None:
            ids.add(str(satellite_id))
    for item in trace_bundle.satellite_catalog:
        satellite_id = item.get("satellite_id")
        if satellite_id is not None:
            ids.add(str(satellite_id))
    return tuple(sorted(ids))


def _request_visible_counts(trace_bundle: RuntimeTraceBundle, workload_generation: Any | None) -> list[int]:
    requests = _workload_requests(workload_generation)
    observations: list[int] = []
    for row in requests:
        time_s = _request_time(row)
        if time_s is None:
            continue
        ingress_id = _request_ingress_id(row)
        satellites = {
            str(window.get("satellite_id"))
            for window in trace_bundle.sgl_windows
            if window.get("satellite_id") is not None
            and _window_active(window, time_s)
            and (ingress_id is None or str(window.get("ingress_id")) == ingress_id)
        }
        if not satellites and ingress_id is not None:
            satellites = {
                str(window.get("satellite_id"))
                for window in trace_bundle.sgl_windows
                if window.get("satellite_id") is not None and _window_active(window, time_s)
            }
        observations.append(len(satellites))
    return observations


def _visible_count_summary(
    *,
    visible_counts: list[int],
    request_counts: list[int],
    access_windows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    counts = visible_counts
    request_weighted = request_counts
    return {
        "sample_count": len(counts),
        "ingress_count": len({str(window.get("ingress_id")) for window in access_windows if window.get("ingress_id") is not None}),
        "access_window_count": len(access_windows),
        "zero_visible_sample_fraction": _zero_fraction(counts),
        "geometry_visible_count_p50": _quantile(counts, 0.50),
        "geometry_visible_count_p95": _quantile(counts, 0.95),
        "geometry_visible_count_max": max(counts, default=0),
        "request_observation_count": len(request_weighted),
        "request_weighted_visible_count_p50": _quantile(request_weighted, 0.50),
        "request_weighted_visible_count_p95": _quantile(request_weighted, 0.95),
        "request_weighted_visible_count_max": max(request_weighted, default=0),
    }


def _access_delay_summary(access_windows: tuple[dict[str, Any], ...]) -> dict[str, float | None]:
    delays = [
        float(value)
        for window in access_windows
        for value in (
            _first_present(window.get("propagation_delay_s"), window.get("propagation_delay_max_s")),
        )
        if value is not None
    ]
    setup = [float(window.get("tau_setup_s")) for window in access_windows if window.get("tau_setup_s") is not None]
    return {
        "p95_s": _quantile(delays, 0.95) if delays else None,
        "setup_p95_s": _quantile(setup, 0.95) if setup else None,
    }


def _constellation_count(trace_bundle: RuntimeTraceBundle, visible_satellite_ids: tuple[str, ...]) -> int:
    return max(len(trace_bundle.satellite_catalog), len(visible_satellite_ids), 1)


def _workload_requests(workload_generation: Any | None) -> tuple[dict[str, Any], ...]:
    if workload_generation is None:
        return ()
    if hasattr(workload_generation, "requests"):
        return tuple(dict(item) for item in getattr(workload_generation, "requests") or ())
    if isinstance(workload_generation, dict):
        return tuple(dict(item) for item in workload_generation.get("requests", ()) if isinstance(item, dict))
    return ()


def _request_time(row: dict[str, Any]) -> float | None:
    value = _first_present(row.get("arrival_time_s"), row.get("time_s"))
    return float(value) if value is not None else None


def _request_ingress_id(row: dict[str, Any]) -> str | None:
    value = _first_present(row.get("ingress_id"), row.get("gateway_id"), row.get("ground_site_id"))
    return str(value) if value is not None else None


def _window_active(window: dict[str, Any], time_s: float) -> bool:
    return float(window.get("start_s", 0.0)) <= time_s < float(window.get("end_s", 0.0))


def _serving_budget_count(serving_throughput: Any | None) -> int | None:
    if serving_throughput is None:
        return None
    if isinstance(serving_throughput, dict):
        value = serving_throughput.get("serving_satellite_budget_count")
    else:
        value = getattr(serving_throughput, "serving_satellite_budget_count", None)
    return int(value) if value is not None else None


def _quantile(values: list[int] | list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _zero_fraction(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(1 for item in values if item <= 0) / len(values)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _visible_count_samples(bundle):
    windows=tuple(bundle.sgl_windows)
    times=sorted({float(row["time_s"]) for row in bundle.ephemeris_samples})
    ingresses=sorted({str(row["ingress_id"]) for row in windows})
    return tuple({"time_s":t,"ingress_id":ingress,"geometry_visible_count":sum(
        str(w["ingress_id"])==ingress and float(w["start_s"])<=t<float(w["end_s"]) for w in windows)}
        for ingress in ingresses for t in times if t<float(bundle.time["end_s"]))
