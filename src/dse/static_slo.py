"""Static workload-SLO proxy for OrbitServe DSE rows.

This module folds canonical workload bundles, serving throughput, and coarse
access/energy summaries into a design-only SLO estimate.  It is intentionally a
bounded static proxy, not a replacement for the trace-backed runtime scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from ..common import to_jsonable
from ..common import PerformanceMetrics, SLOMetrics
from ..orbit import (
    CIRCULAR_GEOMETRY_TRACE,
    IDEAL_FULL_VISIBILITY,
    SURROGATE_ORBIT_TRACE,
    OrbitTraceManifest,
    load_surrogate_trace_bundle,
)


STATIC_SLO_SCOPE = "design_only_surrogate"
STATIC_SLO_WARNING = (
    "static workload-SLO proxy is a design-only surrogate; not STK-calibrated and not valid for paper main results"
)


@dataclass(frozen=True)
class StaticSLOProxyResult:
    evaluated: bool = False
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    slo: SLOMetrics = field(default_factory=SLOMetrics)
    runtime_metrics: dict[str, Any] = field(default_factory=dict)
    timebin_summary: dict[str, Any] = field(default_factory=dict)
    access_summary: dict[str, Any] = field(default_factory=dict)
    energy_summary: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ("static workload-SLO proxy is not evaluated",)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class _AccessEstimate:
    wait_s: float | None
    propagation_delay_s: float
    available: bool
    source: str


def evaluate_static_slo_proxy(
    *,
    workload_generation: Any,
    workload_spec: Any = None,
    serving_throughput: Any,
    simulation_result: Any = None,
    runtime_energy_envelope: Any = None,
    base_dir: str | Path | None = None,
) -> StaticSLOProxyResult:
    """Estimate static served ratio and tail latency for a DSE design row."""

    workload = _mapping(workload_generation)
    requests = tuple(_dict(item) for item in workload.get("requests", ()) if isinstance(item, dict))
    timebins = tuple(_dict(item) for item in workload.get("timebin_demand", ()) if isinstance(item, dict))
    summary = _dict(workload.get("summary"))
    warnings: list[str] = [STATIC_SLO_WARNING]

    if not requests or not timebins:
        return _unevaluated(
            "static workload-SLO proxy requires canonical workload requests and timebin_demand",
            workload_generation=workload_generation,
            serving_throughput=serving_throughput,
        )

    serving = _mapping(serving_throughput)
    if not bool(serving.get("evaluated", False)) or not bool(serving.get("feasible", False)):
        return _unevaluated(
            "static workload-SLO proxy requires an evaluated feasible serving throughput estimate",
            workload_generation=workload_generation,
            serving_throughput=serving_throughput,
        )

    request_capacity = _positive_float(serving.get("request_throughput_per_s"))
    prefill_rate = _positive_float(serving.get("prefill_tokens_per_s"))
    decode_rate = _positive_float(serving.get("decode_tokens_per_s"))
    if request_capacity is None and (prefill_rate is None or decode_rate is None):
        return _unevaluated(
            "static workload-SLO proxy requires request throughput or both prefill/decode token rates",
            workload_generation=workload_generation,
            serving_throughput=serving_throughput,
        )

    trace_bundle, trace_warnings, trace_failure = _load_trace_bundle(simulation_result, base_dir=base_dir)
    if trace_failure is not None:
        return _unevaluated(
            trace_failure,
            workload_generation=workload_generation,
            serving_throughput=serving_throughput,
            extra_warnings=trace_warnings,
        )
    warnings.extend(trace_warnings)
    access_summary = _access_summary(requests, trace_bundle=trace_bundle, simulation_result=simulation_result)
    energy_summary = _energy_summary(trace_bundle=trace_bundle, runtime_energy_envelope=runtime_energy_envelope)
    timebin_summary = _timebin_summary(timebins, request_capacity=request_capacity)

    energy_capacity_factor = _positive_float(energy_summary.get("capacity_factor")) or 1.0
    energy_latency_multiplier = _positive_float(energy_summary.get("latency_multiplier")) or 1.0
    effective_request_capacity = request_capacity * energy_capacity_factor if request_capacity is not None else None
    effective_prefill_rate = prefill_rate * energy_capacity_factor if prefill_rate is not None else None
    effective_decode_rate = decode_rate * energy_capacity_factor if decode_rate is not None else None

    token_refs = _token_refs(serving, summary, workload_spec, requests)
    base_ttft_s = _nonnegative_float(serving.get("ttft_s"))
    base_e2e_s = _nonnegative_float(serving.get("e2e_s"))
    decode_per_output_token_s = _decode_latency_per_token(
        base_ttft_s=base_ttft_s,
        base_e2e_s=base_e2e_s,
        output_tokens_p95=token_refs["output_tokens_p95"],
        decode_rate=decode_rate,
    )

    available_at = 0.0
    records: list[dict[str, Any]] = []
    for request in sorted(requests, key=lambda item: (float(item.get("arrival_time_s", 0.0)), str(item.get("request_id", "")))):
        arrival = _float(request.get("arrival_time_s"), 0.0)
        access = _access_estimate(request, trace_bundle=trace_bundle)
        if not access.available or access.wait_s is None:
            records.append(
                {
                    "request_id": request.get("request_id"),
                    "arrival_time_s": arrival,
                    "prompt_tokens": max(0, _int(_first_present(request.get("prompt_tokens"), request.get("new_user_tokens")), 0)),
                    "output_tokens": max(0, _int(request.get("output_tokens"), 0)),
                    "accepted": False,
                    "served": False,
                    "timeout": False,
                    "rejected": True,
                    "reject_reason": "no_access_window",
                    "ttft_s": None,
                    "e2e_s": None,
                    "queue_wait_s": None,
                    "access_wait_s": None,
                    "access_delay_s": None,
                }
            )
            continue

        prompt_tokens = max(0, _int(_first_present(request.get("prompt_tokens"), request.get("new_user_tokens")), 0))
        output_tokens = max(0, _int(request.get("output_tokens"), 0))
        service_time_s = _service_time_s(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            request_capacity=effective_request_capacity,
            prefill_rate=effective_prefill_rate,
            decode_rate=effective_decode_rate,
            token_refs=token_refs,
        )
        if service_time_s is None:
            records.append(
                {
                    "request_id": request.get("request_id"),
                    "arrival_time_s": arrival,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "accepted": False,
                    "served": False,
                    "timeout": False,
                    "rejected": True,
                    "reject_reason": "capacity_unavailable",
                    "ttft_s": None,
                    "e2e_s": None,
                    "queue_wait_s": None,
                    "access_wait_s": access.wait_s,
                    "access_delay_s": None,
                }
            )
            continue

        ready_time = arrival + access.wait_s
        start_time = max(ready_time, available_at)
        queue_wait_s = max(0.0, start_time - ready_time)
        available_at = start_time + service_time_s

        ttft_compute_s = _ttft_compute_s(
            prompt_tokens=prompt_tokens,
            base_ttft_s=base_ttft_s,
            prefill_rate=prefill_rate,
            prompt_tokens_p95=token_refs["prompt_tokens_p95"],
            latency_multiplier=energy_latency_multiplier,
        )
        decode_compute_s = output_tokens * decode_per_output_token_s * energy_latency_multiplier
        access_delay_s = access.wait_s + access.propagation_delay_s
        ttft_s = access_delay_s + queue_wait_s + ttft_compute_s
        e2e_s = ttft_s + decode_compute_s + access.propagation_delay_s
        ttft_deadline = _optional_float(request.get("ttft_deadline_s"))
        e2e_deadline = _optional_float(request.get("e2e_deadline_s"))
        ttft_ok = True if ttft_deadline is None else ttft_s <= ttft_deadline
        e2e_ok = True if e2e_deadline is None else e2e_s <= e2e_deadline
        served = ttft_ok and e2e_ok
        records.append(
            {
                "request_id": request.get("request_id"),
                "arrival_time_s": arrival,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "accepted": True,
                "served": served,
                "timeout": not served,
                "rejected": False,
                "reject_reason": None,
                "ttft_s": ttft_s,
                "e2e_s": e2e_s,
                "queue_wait_s": queue_wait_s,
                "access_wait_s": access.wait_s,
                "access_delay_s": access_delay_s,
                "slo_class": request.get("slo_class"),
            }
        )

    metrics = _aggregate_records(records, time_horizon_s=_horizon_s(requests, timebins, workload_spec))
    warnings.extend(access_summary.get("warnings", ()))
    warnings.extend(energy_summary.get("warnings", ()))
    warnings.extend(timebin_summary.get("warnings", ()))
    warnings = list(dict.fromkeys(str(warning) for warning in warnings))

    performance = PerformanceMetrics(
        evaluated=True,
        request_throughput_per_s=metrics["useful_request_throughput_per_s"],
        token_throughput_per_s=metrics["useful_token_throughput_per_s"],
        ttft_p95_s=metrics["ttft_p95_s"],
        ttft_p99_s=metrics["ttft_p99_s"],
        e2e_p95_s=metrics["e2e_p95_s"],
        e2e_p99_s=metrics["e2e_p99_s"],
        queue_delay_p95_s=metrics["queue_delay_p95_s"],
        link_delay_p95_s=metrics["access_delay_p95_s"],
        prefill_tokens_per_s=effective_prefill_rate,
        decode_tokens_per_s=effective_decode_rate,
        kv_cache_capacity_tokens=_optional_int(serving.get("kv_cache_capacity_tokens")),
        satellite_count_used=_optional_int(serving.get("satellite_count_used")),
        feasible=metrics["served_ratio"] is not None and metrics["served_ratio"] > 0.0,
        bottleneck=_static_bottleneck(metrics, timebin_summary, access_summary, energy_summary),
        details={
            "analysis_scope": STATIC_SLO_SCOPE,
            "valid_for_paper_main_results": False,
            "trace_model": access_summary.get("trace_model"),
            "static_slo_proxy": {
                "runtime_metrics": metrics,
                "timebin_summary": timebin_summary,
                "access_summary": access_summary,
                "energy_summary": energy_summary,
                "token_refs": token_refs,
            },
        },
        warnings=tuple(warnings),
    )
    slo = SLOMetrics(
        evaluated=True,
        served_ratio=metrics["served_ratio"],
        rejected_ratio=metrics["rejected_ratio"],
        timeout_ratio=metrics["timeout_ratio"],
        energy_violation_count=None,
        isl_congestion_p95=None,
        sgl_congestion_p95=None,
        warnings=tuple(warnings),
    )
    return StaticSLOProxyResult(
        evaluated=True,
        performance=performance,
        slo=slo,
        runtime_metrics=metrics,
        timebin_summary=timebin_summary,
        access_summary=access_summary,
        energy_summary=energy_summary,
        provenance={
            "source": "orbitserve.static_slo.evaluate_static_slo_proxy",
            "analysis_scope": STATIC_SLO_SCOPE,
            "valid_for_paper_main_results": False,
            "workload_request_count": len(requests),
            "timebin_count": len(timebins),
        },
        warnings=tuple(warnings),
    )


def _unevaluated(
    reason: str,
    *,
    workload_generation: Any,
    serving_throughput: Any,
    extra_warnings: tuple[str, ...] = (),
) -> StaticSLOProxyResult:
    warnings = tuple(dict.fromkeys((reason, *extra_warnings, STATIC_SLO_WARNING)))
    return StaticSLOProxyResult(
        evaluated=False,
        performance=PerformanceMetrics(warnings=warnings),
        slo=SLOMetrics(warnings=warnings),
        provenance={
            "source": "orbitserve.static_slo.evaluate_static_slo_proxy",
            "analysis_scope": STATIC_SLO_SCOPE,
            "valid_for_paper_main_results": False,
            "workload_configured": bool(_mapping(workload_generation)),
            "serving_evaluated": bool(_mapping(serving_throughput).get("evaluated", False)),
        },
        warnings=warnings,
    )


def _load_trace_bundle(
    simulation_result: Any,
    *,
    base_dir: str | Path | None,
) -> tuple[Any | None, tuple[str, ...], str | None]:
    payload = _mapping(simulation_result)
    if not payload:
        return (
            None,
            ("static SLO proxy has no simulation result; using legacy full-visibility, no-energy-derating assumptions",),
            None,
        )
    manifest_payload = payload.get("trace_manifest") or payload
    manifest = OrbitTraceManifest.from_value(manifest_payload)
    payload_warning_value = payload.get("warnings", ())
    payload_warnings = (
        (payload_warning_value,)
        if isinstance(payload_warning_value, str)
        else tuple(str(warning) for warning in payload_warning_value or ())
    )
    trace_warnings = tuple(
        dict.fromkeys(
            (
                *payload_warnings,
                *manifest.warnings,
            )
        )
    )
    if "trace_evaluated" in payload and payload.get("trace_evaluated") is not True:
        reason = (
            "static SLO proxy is not evaluated because simulation_result.trace_evaluated is not true "
            "and the configured trace did not produce an evaluated trace manifest"
        )
        return None, trace_warnings, reason
    if "trace_model" in payload and not _optional_str(payload.get("trace_model")):
        reason = "static SLO proxy is not evaluated because simulation_result.trace_model is empty"
        return None, trace_warnings, reason
    top_level_trace_model = _optional_str(payload.get("trace_model"))
    if top_level_trace_model is not None and top_level_trace_model != manifest.trace_model:
        reason = (
            "static SLO proxy is not evaluated because simulation_result.trace_model "
            f"({top_level_trace_model}) disagrees with trace_manifest.trace_model ({manifest.trace_model})"
        )
        return None, trace_warnings, reason
    if not manifest.evaluated:
        reason = (
            "static SLO proxy is not evaluated because configured "
            f"trace_model={manifest.trace_model} did not produce an evaluated trace manifest"
        )
        return None, trace_warnings, reason
    if manifest.trace_model == IDEAL_FULL_VISIBILITY:
        return None, ("static SLO proxy uses explicitly requested ideal_full_visibility access summary",), None
    if manifest.trace_model not in {SURROGATE_ORBIT_TRACE, CIRCULAR_GEOMETRY_TRACE}:
        reason = f"static SLO proxy is not evaluated because trace_model={manifest.trace_model} is unsupported"
        return None, trace_warnings, reason
    try:
        trace_bundle = load_surrogate_trace_bundle(manifest, base_dir=base_dir)
    except (OSError, ValueError) as exc:
        reason = (
            "static SLO proxy is not evaluated because the evaluated "
            f"trace_model={manifest.trace_model} manifest could not be loaded: {exc}"
        )
        return None, trace_warnings, reason
    trace_failure = _trace_bundle_failure(trace_bundle)
    if trace_failure is not None:
        return None, tuple(getattr(trace_bundle, "warnings", ()) or trace_warnings), trace_failure
    return trace_bundle, (), None


def _trace_bundle_failure(trace_bundle: Any) -> str | None:
    trace_model = getattr(trace_bundle, "trace_model", None)
    access_trace = getattr(trace_bundle, "access_trace", None)
    if not isinstance(access_trace, dict) or "access_windows" not in access_trace:
        return (
            "static SLO proxy is not evaluated because loaded "
            f"trace_model={trace_model} bundle is missing the access_windows field"
        )
    if not isinstance(access_trace.get("access_windows"), list):
        return (
            "static SLO proxy is not evaluated because loaded "
            f"trace_model={trace_model} bundle access_windows field is not a list"
        )
    for index, window in enumerate(access_trace["access_windows"]):
        if not isinstance(window, dict):
            return (
                "static SLO proxy is not evaluated because loaded "
                f"trace_model={trace_model} access_windows[{index}] is not an object"
            )
        start_s = _nonnegative_float(window.get("start_s"))
        end_s = _nonnegative_float(window.get("end_s"))
        propagation_delay_s = _nonnegative_float(window.get("propagation_delay_s"))
        if (
            _optional_str(window.get("ingress_id")) is None
            or start_s is None
            or end_s is None
            or end_s <= start_s
            or propagation_delay_s is None
        ):
            return (
                "static SLO proxy is not evaluated because loaded "
                f"trace_model={trace_model} access_windows[{index}] lacks valid ingress_id, "
                "start_s/end_s, or propagation_delay_s"
            )

    energy_trace = getattr(trace_bundle, "energy_trace", None)
    if not isinstance(energy_trace, dict) or not isinstance(energy_trace.get("energy_windows"), list):
        return (
            "static SLO proxy is not evaluated because loaded "
            f"trace_model={trace_model} bundle is missing a list-valued energy_windows field"
        )
    energy_windows = energy_trace["energy_windows"]
    if not energy_windows:
        return (
            "static SLO proxy is not evaluated because loaded "
            f"trace_model={trace_model} bundle has no energy_windows"
        )
    if any(not isinstance(window, dict) for window in energy_windows):
        return (
            "static SLO proxy is not evaluated because loaded "
            f"trace_model={trace_model} bundle contains a non-object energy window"
        )
    if any("energy_unavailable_reason" in window for window in energy_windows):
        return (
            "static SLO proxy is not evaluated because loaded "
            f"trace_model={trace_model} bundle contains unavailable energy_windows"
        )
    for index, window in enumerate(energy_windows):
        start_s = _nonnegative_float(window.get("start_s"))
        end_s = _nonnegative_float(window.get("end_s"))
        if (
            _optional_str(window.get("satellite_id")) is None
            or start_s is None
            or end_s is None
            or end_s <= start_s
            or _positive_float(window.get("dvfs_multiplier")) is None
        ):
            return (
                "static SLO proxy is not evaluated because loaded "
                f"trace_model={trace_model} energy_windows[{index}] lacks valid satellite_id, "
                "start_s/end_s, or dvfs_multiplier"
            )
    return None


def _access_summary(requests: tuple[dict[str, Any], ...], *, trace_bundle: Any | None, simulation_result: Any) -> dict[str, Any]:
    if trace_bundle is None:
        trace_payload = _mapping(simulation_result)
        trace_model = _optional_str(_first_present(trace_payload.get("trace_model"), _dict(trace_payload.get("trace_manifest")).get("trace_model")))
        return {
            "evaluated": False,
            "trace_model": trace_model or "not_evaluated",
            "assumption": "full_visibility_zero_access_wait",
            "request_access_coverage_ratio": 1.0,
            "access_wait_p95_s": 0.0,
            "access_wait_p99_s": 0.0,
            "propagation_delay_p95_s": 0.0,
            "warnings": ("access summary unavailable; static proxy assumes full visibility",),
        }
    waits: list[float | None] = []
    propagation_delays: list[float | None] = []
    covered = 0
    for request in requests:
        estimate = _access_estimate(request, trace_bundle=trace_bundle)
        if estimate.available:
            covered += 1
            waits.append(estimate.wait_s)
            propagation_delays.append(estimate.propagation_delay_s)
        else:
            waits.append(None)
    total = len(requests)
    return {
        "evaluated": True,
        "trace_model": getattr(trace_bundle, "trace_model", None),
        "scope": getattr(trace_bundle, "scope", None),
        "access_window_count": len(getattr(trace_bundle, "access_windows", ()) or ()),
        "request_access_coverage_ratio": covered / total if total else None,
        "access_wait_p95_s": _percentile(waits, 0.95),
        "access_wait_p99_s": _percentile(waits, 0.99),
        "propagation_delay_p95_s": _percentile(propagation_delays, 0.95),
        "warnings": ("access summary is a static window lookup, not scheduler path selection",),
    }


def _access_estimate(request: dict[str, Any], *, trace_bundle: Any | None) -> _AccessEstimate:
    if trace_bundle is None:
        return _AccessEstimate(wait_s=0.0, propagation_delay_s=0.0, available=True, source="assumed_full_visibility")
    ingress_id = _optional_str(_first_present(request.get("ingress_id"), request.get("gateway_id"), request.get("region_id")))
    if ingress_id is None:
        return _AccessEstimate(wait_s=None, propagation_delay_s=0.0, available=False, source="missing_ingress_id")
    arrival = _float(request.get("arrival_time_s"), 0.0)
    windows = [
        _dict(window)
        for window in getattr(trace_bundle, "access_windows", ()) or ()
        if str(window.get("ingress_id")) == ingress_id
    ]
    current = [
        window
        for window in windows
        if _float(window.get("start_s"), 0.0) <= arrival < _float(window.get("end_s"), 0.0)
    ]
    if current:
        window = min(current, key=lambda item: _float(item.get("propagation_delay_s"), 0.0))
        return _AccessEstimate(
            wait_s=0.0,
            propagation_delay_s=_float(window.get("propagation_delay_s"), 0.0),
            available=True,
            source="access_window_at_arrival",
        )
    future = [
        window
        for window in windows
        if _float(window.get("start_s"), 0.0) >= arrival
    ]
    if not future:
        return _AccessEstimate(wait_s=None, propagation_delay_s=0.0, available=False, source="no_future_access_window")
    window = min(future, key=lambda item: (_float(item.get("start_s"), 0.0), _float(item.get("propagation_delay_s"), 0.0)))
    return _AccessEstimate(
        wait_s=max(0.0, _float(window.get("start_s"), 0.0) - arrival),
        propagation_delay_s=_float(window.get("propagation_delay_s"), 0.0),
        available=True,
        source="next_access_window",
    )


def _energy_summary(*, trace_bundle: Any | None, runtime_energy_envelope: Any) -> dict[str, Any]:
    envelope = _mapping(runtime_energy_envelope)
    if trace_bundle is None:
        return {
            "evaluated": bool(envelope.get("evaluated", False)),
            "capacity_factor": 1.0,
            "latency_multiplier": 1.0,
            "source": "runtime_energy_envelope" if envelope else "not_configured",
            "warnings": ("energy summary unavailable; static proxy applies no energy derating",),
        }
    windows = tuple(_dict(item) for item in getattr(trace_bundle, "energy_windows", ()) or ())
    if not windows:
        return {
            "evaluated": False,
            "capacity_factor": 1.0,
            "latency_multiplier": 1.0,
            "source": getattr(trace_bundle, "trace_model", None),
            "warnings": ("energy trace has no energy_windows; static proxy applies no energy derating",),
        }
    if any("energy_unavailable_reason" in window for window in windows):
        return {
            "evaluated": False,
            "capacity_factor": 1.0,
            "latency_multiplier": 1.0,
            "source": getattr(trace_bundle, "trace_model", None),
            "warnings": ("energy trace lacks runtime energy envelope; static proxy applies no energy derating",),
        }
    weighted_duration = 0.0
    weighted_multiplier = 0.0
    sunlit_duration = 0.0
    for window in windows:
        duration = _duration(window)
        if duration <= 0:
            continue
        multiplier = _positive_float(window.get("dvfs_multiplier")) or 1.0
        weighted_duration += duration
        weighted_multiplier += duration * multiplier
        if bool(window.get("sunlit", False)):
            sunlit_duration += duration
    latency_multiplier = weighted_multiplier / weighted_duration if weighted_duration > 0 else 1.0
    capacity_factor = 1.0 / latency_multiplier if latency_multiplier > 0 else 1.0
    return {
        "evaluated": True,
        "source": getattr(trace_bundle, "trace_model", None),
        "energy_window_count": len(windows),
        "sunlit_fraction": sunlit_duration / weighted_duration if weighted_duration > 0 else None,
        "shadow_fraction": 1.0 - sunlit_duration / weighted_duration if weighted_duration > 0 else None,
        "latency_multiplier": latency_multiplier,
        "capacity_factor": capacity_factor,
        "warnings": ("energy summary uses duration-weighted DVFS multiplier, not scheduler SOC integration",),
    }


def _timebin_summary(timebins: tuple[dict[str, Any], ...], *, request_capacity: float | None) -> dict[str, Any]:
    rates: list[float] = []
    total_requests = 0
    overloaded = 0
    for row in timebins:
        duration = max(0.0, _float(row.get("time_bin_end_s"), 0.0) - _float(row.get("time_bin_start_s"), 0.0))
        request_count = max(0, _int(row.get("request_count"), 0))
        total_requests += request_count
        rate = request_count / duration if duration > 0 else 0.0
        rates.append(rate)
        if request_capacity is not None and rate > request_capacity:
            overloaded += 1
    peak_rate = max(rates) if rates else None
    mean_rate = sum(rates) / len(rates) if rates else None
    return {
        "evaluated": True,
        "timebin_count": len(timebins),
        "request_count": total_requests,
        "mean_request_rate_per_s": mean_rate,
        "peak_request_rate_per_s": peak_rate,
        "peak_to_capacity_ratio": peak_rate / request_capacity if peak_rate is not None and request_capacity else None,
        "overloaded_timebin_fraction": overloaded / len(timebins) if timebins else None,
        "warnings": ("timebin_demand is used for offered-load summary; request arrivals drive latency estimates",),
    }


def _aggregate_records(records: list[dict[str, Any]], *, time_horizon_s: float | None) -> dict[str, Any]:
    total = len(records)
    accepted = [row for row in records if row.get("accepted")]
    served = [row for row in records if row.get("served")]
    rejected = [row for row in records if row.get("rejected")]
    timeout = [row for row in records if row.get("timeout")]
    completed = [row for row in records if row.get("ttft_s") is not None and row.get("e2e_s") is not None]
    served_tokens = sum(
        _int(row.get("prompt_tokens"), 0) + _int(row.get("output_tokens"), 0)
        for row in served
    )
    horizon = time_horizon_s or _record_horizon(records)
    return {
        "analysis_scope": STATIC_SLO_SCOPE,
        "valid_for_paper_main_results": False,
        "total_request_count": total,
        "accepted_request_count": len(accepted),
        "served_request_count": len(served),
        "rejected_request_count": len(rejected),
        "timeout_request_count": len(timeout),
        "served_ratio": len(served) / total if total else None,
        "accepted_ratio": len(accepted) / total if total else None,
        "rejected_ratio": len(rejected) / total if total else None,
        "timeout_ratio": len(timeout) / total if total else None,
        "ttft_p50_s": _percentile([row.get("ttft_s") for row in completed], 0.50),
        "ttft_p95_s": _percentile([row.get("ttft_s") for row in completed], 0.95),
        "ttft_p99_s": _percentile([row.get("ttft_s") for row in completed], 0.99),
        "e2e_p50_s": _percentile([row.get("e2e_s") for row in completed], 0.50),
        "e2e_p95_s": _percentile([row.get("e2e_s") for row in completed], 0.95),
        "e2e_p99_s": _percentile([row.get("e2e_s") for row in completed], 0.99),
        "queue_delay_p95_s": _percentile([row.get("queue_wait_s") for row in completed], 0.95),
        "access_wait_p95_s": _percentile([row.get("access_wait_s") for row in completed], 0.95),
        "access_delay_p95_s": _percentile([row.get("access_delay_s") for row in completed], 0.95),
        "useful_request_throughput_per_s": len(served) / horizon if horizon and horizon > 0 else None,
        "useful_token_throughput_per_s": served_tokens / horizon if horizon and horizon > 0 else None,
        "tail_latency_population": "accepted_completed_requests",
    }


def _service_time_s(
    *,
    prompt_tokens: int,
    output_tokens: int,
    request_capacity: float | None,
    prefill_rate: float | None,
    decode_rate: float | None,
    token_refs: dict[str, int],
) -> float | None:
    if prefill_rate and decode_rate:
        return prompt_tokens / prefill_rate + output_tokens / decode_rate
    if request_capacity:
        prompt_norm = prompt_tokens / max(1, token_refs["prompt_tokens_p50"])
        output_norm = output_tokens / max(1, token_refs["output_tokens_p50"])
        return max(0.0, (prompt_norm + output_norm) / (2.0 * request_capacity))
    return None


def _ttft_compute_s(
    *,
    prompt_tokens: int,
    base_ttft_s: float | None,
    prefill_rate: float | None,
    prompt_tokens_p95: int,
    latency_multiplier: float,
) -> float:
    if base_ttft_s is not None and prompt_tokens_p95 > 0:
        return max(0.0, base_ttft_s * prompt_tokens / prompt_tokens_p95 * latency_multiplier)
    if prefill_rate:
        return prompt_tokens / prefill_rate * latency_multiplier
    return 0.0


def _decode_latency_per_token(
    *,
    base_ttft_s: float | None,
    base_e2e_s: float | None,
    output_tokens_p95: int,
    decode_rate: float | None,
) -> float:
    if base_ttft_s is not None and base_e2e_s is not None and base_e2e_s >= base_ttft_s and output_tokens_p95 > 0:
        return (base_e2e_s - base_ttft_s) / output_tokens_p95
    if decode_rate:
        return 1.0 / decode_rate
    return 0.0


def _token_refs(serving: dict[str, Any], summary: dict[str, Any], workload_spec: Any, requests: tuple[dict[str, Any], ...]) -> dict[str, int]:
    provenance_tokens = _dict(_dict(serving.get("provenance")).get("token_distribution"))
    workload_tokens = _mapping(getattr(workload_spec, "token_distribution", None))
    summary_tokens = _dict(summary.get("token_stats"))
    prompt_stats = _dict(summary_tokens.get("prompt_tokens"))
    output_stats = _dict(summary_tokens.get("output_tokens"))
    prompt_values = [max(1, _int(row.get("prompt_tokens"), 1)) for row in requests]
    output_values = [max(1, _int(row.get("output_tokens"), 1)) for row in requests]
    return {
        "prompt_tokens_p50": max(1, _int(_first_present(provenance_tokens.get("prompt_tokens_p50"), workload_tokens.get("prompt_tokens_p50"), prompt_stats.get("p50"), _percentile(prompt_values, 0.50)), 1)),
        "prompt_tokens_p95": max(1, _int(_first_present(provenance_tokens.get("prompt_tokens_p95"), workload_tokens.get("prompt_tokens_p95"), prompt_stats.get("p95"), _percentile(prompt_values, 0.95)), 1)),
        "output_tokens_p50": max(1, _int(_first_present(provenance_tokens.get("output_tokens_p50"), workload_tokens.get("output_tokens_p50"), output_stats.get("p50"), _percentile(output_values, 0.50)), 1)),
        "output_tokens_p95": max(1, _int(_first_present(provenance_tokens.get("output_tokens_p95"), workload_tokens.get("output_tokens_p95"), output_stats.get("p95"), _percentile(output_values, 0.95)), 1)),
    }


def _static_bottleneck(
    metrics: dict[str, Any],
    timebin_summary: dict[str, Any],
    access_summary: dict[str, Any],
    energy_summary: dict[str, Any],
) -> str | None:
    if metrics.get("rejected_ratio") and metrics["rejected_ratio"] > 0:
        return "access"
    if metrics.get("timeout_ratio") and metrics["timeout_ratio"] > 0:
        return "slo_tail_latency"
    if timebin_summary.get("peak_to_capacity_ratio") and timebin_summary["peak_to_capacity_ratio"] > 1:
        return "timebin_offered_load"
    if energy_summary.get("capacity_factor") and energy_summary["capacity_factor"] < 1:
        return "energy_dvfs"
    if access_summary.get("request_access_coverage_ratio") not in (None, 1.0):
        return "access_coverage"
    return "none"


def _horizon_s(requests: tuple[dict[str, Any], ...], timebins: tuple[dict[str, Any], ...], workload_spec: Any) -> float | None:
    configured = _optional_float(getattr(workload_spec, "duration_s", None))
    if configured and configured > 0:
        return configured
    max_bin = max((_optional_float(row.get("time_bin_end_s")) or 0.0 for row in timebins), default=0.0)
    max_arrival = max((_optional_float(row.get("arrival_time_s")) or 0.0 for row in requests), default=0.0)
    return max(max_bin, max_arrival) or None


def _record_horizon(records: list[dict[str, Any]]) -> float | None:
    arrivals = [_optional_float(row.get("arrival_time_s")) for row in records]
    clean = [item for item in arrivals if item is not None]
    return max(clean) - min(clean) if len(clean) > 1 else None


def _duration(window: dict[str, Any]) -> float:
    if window.get("duration_s") is not None:
        return max(0.0, _float(window.get("duration_s"), 0.0))
    return max(0.0, _float(window.get("end_s"), 0.0) - _float(window.get("start_s"), 0.0))


def _percentile(values: list[float | int | None], percentile: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[int(position)]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return to_jsonable(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = to_jsonable(value.to_dict())
        return converted if isinstance(converted, dict) else {}
    converted = to_jsonable(value)
    return converted if isinstance(converted, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_float(value: Any) -> float | None:
    parsed = _optional_float(value)
    return parsed if parsed is not None and parsed > 0.0 else None


def _nonnegative_float(value: Any) -> float | None:
    parsed = _optional_float(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int) -> int:
    parsed = _optional_int(value)
    return default if parsed is None else parsed


def _float(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


__all__ = [
    "STATIC_SLO_SCOPE",
    "STATIC_SLO_WARNING",
    "StaticSLOProxyResult",
    "evaluate_static_slo_proxy",
]
