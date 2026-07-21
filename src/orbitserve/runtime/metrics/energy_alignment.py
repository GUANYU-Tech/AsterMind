"""Post-runtime audit for workload-derived energy sizing.

This module is deliberately read-only with respect to construction sizing.  A
planned demand is produced from the offered workload before cost evaluation;
the helpers here only compare that immutable plan with request-event ledger
rows emitted by runtime.  Runtime admission outcomes must therefore never
shrink the solar array or battery selected by the cost model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Mapping

from ...common import to_jsonable


DEFAULT_ALIGNMENT_TOLERANCE_FRACTION = 0.05


@dataclass(frozen=True)
class EnergyAlignmentResult:
    evaluated: bool = False
    planned_energy: dict[str, Any] = field(default_factory=dict)
    runtime_energy: dict[str, Any] = field(default_factory=dict)
    difference_fraction: float | None = None
    tolerance_fraction: float = DEFAULT_ALIGNMENT_TOLERANCE_FRACTION
    target_served_ratio: float | None = None
    runtime_completion_ratio: float | None = None
    target_served_ratio_met: bool | None = None
    coefficients_match: bool | None = None
    sunlight_fraction_match: bool | None = None
    eclipse_duration_sufficient: bool | None = None
    sunlight_contract: dict[str, Any] = field(default_factory=dict)
    consistent: bool = False
    requires_resizing: bool = False
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def evaluate_energy_alignment(
    planned_energy_demand: Any,
    runtime_performance: Any,
    *,
    runtime_energy_envelope: Any = None,
    runtime_sunlight_summary: Any = None,
    tolerance_fraction: float = DEFAULT_ALIGNMENT_TOLERANCE_FRACTION,
) -> EnergyAlignmentResult:
    """Compare one offered-workload energy plan with runtime request ledgers."""

    tolerance = _nonnegative_finite(tolerance_fraction, "tolerance_fraction")
    planned = _mapping(planned_energy_demand)
    runtime = _mapping(runtime_performance)
    planned_evaluated = bool(planned.get("evaluated", bool(planned)))
    planned_compute_j = _first_finite(
        planned.get("planned_compute_energy_j"),
        planned.get("compute_energy_j"),
        planned.get("total_compute_energy_j"),
        _nested(planned, "energy", "compute_j"),
    )
    planned_link_j = _first_finite(
        planned.get("planned_link_energy_j"),
        planned.get("link_energy_j"),
        planned.get("total_link_energy_j"),
        _nested(planned, "energy", "link_j"),
    )
    planned_dynamic_j = _first_finite(
        planned.get("planned_dynamic_energy_j"),
        planned.get("dynamic_energy_j"),
        planned.get("total_dynamic_energy_j"),
        _nested(planned, "energy", "dynamic_j"),
    )
    if planned_dynamic_j is None and planned_compute_j is not None and planned_link_j is not None:
        planned_dynamic_j = planned_compute_j + planned_link_j
    planned_per_satellite_dynamic_j = _first_finite(
        planned.get("per_satellite_dynamic_energy_j")
    )
    planned_maximum_request_j = _first_finite(
        planned.get("maximum_request_dynamic_energy_j")
    )
    planned_per_satellite_buffer_j = _first_finite(
        planned.get("per_satellite_dynamic_buffer_energy_j")
    )
    if planned_per_satellite_buffer_j is None:
        candidates = tuple(
            value
            for value in (
                planned_per_satellite_dynamic_j,
                planned_maximum_request_j,
            )
            if value is not None
        )
        planned_per_satellite_buffer_j = max(candidates) if candidates else None

    warnings: list[str] = []
    if not planned_evaluated:
        warnings.append("planned energy demand is not evaluated")
    if planned_dynamic_j is None or planned_dynamic_j < 0:
        warnings.append("planned dynamic energy is missing or invalid")

    planned_coefficients = _planned_coefficients(planned)
    records = _runtime_request_records(runtime_performance, runtime)
    runtime_energy_ledger_expected = _runtime_energy_ledger_expected(runtime)
    actual = _aggregate_request_event_energy(
        records,
        require_request_event_ledger=runtime_energy_ledger_expected,
    )
    offered_request_count = _first_int(
        planned.get("offered_request_count"),
        planned.get("request_count"),
        _nested(planned, "workload", "request_count"),
    )
    zero_workload = offered_request_count == 0
    zero_planned_dynamic = planned_dynamic_j == 0
    ledger_available = actual["request_event_count"] > 0 or (
        zero_planned_dynamic and actual["runtime_dynamic_energy_j"] == 0
    )
    if actual["invalid_ledger_entry_count"]:
        warnings.append("runtime energy ledger contains invalid request-event energy values")
    if actual["missing_required_ledger_record_count"]:
        warnings.append(
            "one or more satellite-completed requests have no request-event energy ledger"
        )
    if not ledger_available:
        warnings.append("runtime request-event energy ledger is unavailable")

    runtime_evaluated = bool(runtime.get("evaluated", getattr(runtime_performance, "evaluated", False)))
    runtime_completion_ratio = _first_finite(
        _nested(runtime, "runtime_metrics", "completion_ratio"),
    )
    missing_runtime_request_record_count = 0
    if (
        runtime_energy_ledger_expected
        and runtime_completion_ratio is not None
        and runtime_completion_ratio > 0.0
    ):
        if offered_request_count is not None:
            missing_runtime_request_record_count = max(
                0,
                offered_request_count - len(records),
            )
        elif not records:
            missing_runtime_request_record_count = 1
    actual["missing_runtime_request_record_count"] = (
        missing_runtime_request_record_count
    )
    if missing_runtime_request_record_count:
        warnings.append(
            "runtime reports completed satellite work but its request-record collection is incomplete"
        )
    request_record_completion_ratio = None
    if offered_request_count is not None and offered_request_count > 0:
        request_record_completion_ratio = (
            actual["completed_request_count"] / offered_request_count
        )
    runtime_request_record_mismatch = False
    if (
        runtime_energy_ledger_expected
        and runtime_completion_ratio is not None
        and not zero_workload
    ):
        if request_record_completion_ratio is not None:
            runtime_request_record_mismatch = not math.isclose(
                runtime_completion_ratio,
                request_record_completion_ratio,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        elif runtime_completion_ratio > 0.0 and actual["completed_request_count"] == 0:
            runtime_request_record_mismatch = True
    actual["request_record_completion_ratio"] = request_record_completion_ratio
    actual["runtime_request_record_mismatch"] = runtime_request_record_mismatch
    if runtime_request_record_mismatch:
        warnings.append(
            "runtime completed-ratio metrics disagree with the request-record collection"
        )
    target_served_ratio = _target_served_ratio(planned)
    target_met = None
    if target_served_ratio is not None:
        target_met = True if zero_workload else (
            runtime_completion_ratio is not None
            and runtime_completion_ratio + 1e-12 >= target_served_ratio
        )
        if not target_met:
            warnings.append(
                "runtime completed ratio did not meet the sizing target; planned energy remains unchanged"
            )

    runtime_coefficients = _runtime_coefficients(runtime_energy_envelope)
    envelope_coefficients_match = _coefficient_match(planned_coefficients, runtime_coefficients)
    ledger_coefficient_audit = _audit_runtime_ledger_coefficients(
        records,
        planned_coefficients,
        allow_empty=zero_planned_dynamic,
    )
    actual["coefficient_audit"] = ledger_coefficient_audit
    if envelope_coefficients_match is False or ledger_coefficient_audit["coefficients_match"] is False:
        coefficients_match = False
    elif (
        envelope_coefficients_match is True
        and ledger_coefficient_audit["coefficients_match"] is True
    ):
        coefficients_match = True
    else:
        coefficients_match = None
    if coefficients_match is False:
        warnings.append("planned and runtime energy coefficients do not match")
    elif coefficients_match is None:
        warnings.append("runtime coefficient provenance is unavailable for comparison")

    expected_sunlit_fraction = _first_finite(
        _nested(_mapping(runtime_energy_envelope), "provenance", "hardware_power", "sunlit_fraction"),
        1.0 - float(_nested(_mapping(runtime_energy_envelope), "trace_options", "shadow_fraction"))
        if _finite_nonnegative_or_none(
            _nested(_mapping(runtime_energy_envelope), "trace_options", "shadow_fraction")
        )
        is not None
        else None,
    )
    sunlight_summary = _mapping(runtime_sunlight_summary)
    sunlight_summary_evaluated = bool(sunlight_summary.get("evaluated", False))
    actual_sunlit_fraction = _first_finite(
        sunlight_summary.get("actual_sunlit_fraction"),
        sunlight_summary.get("sunlit_fraction"),
    )
    sunlight_difference = None
    sunlight_fraction_match = None
    if (
        expected_sunlit_fraction is not None
        and 0.0 <= expected_sunlit_fraction <= 1.0
        and actual_sunlit_fraction is not None
        and 0.0 <= actual_sunlit_fraction <= 1.0
    ):
        sunlight_difference = abs(actual_sunlit_fraction - expected_sunlit_fraction)
        sunlight_fraction_match = sunlight_difference <= tolerance + 1e-12
        if not sunlight_fraction_match:
            warnings.append(
                "runtime sunlight fraction differs from the cost sizing assumption"
            )
    cost_eclipse_duration_s = _first_finite(
        (
            float(
                _nested(
                    _mapping(runtime_energy_envelope),
                    "provenance",
                    "hardware_power",
                    "eclipse_duration_min",
                )
            )
            * 60.0
        )
        if _finite_nonnegative_or_none(
            _nested(
                _mapping(runtime_energy_envelope),
                "provenance",
                "hardware_power",
                "eclipse_duration_min",
            )
        )
        is not None
        else None
    )
    runtime_maximum_shadow_s = _first_finite(
        sunlight_summary.get("maximum_shadow_window_s")
    )
    eclipse_duration_sufficient = None
    if cost_eclipse_duration_s is not None and runtime_maximum_shadow_s is not None:
        eclipse_duration_sufficient = bool(
            runtime_maximum_shadow_s
            <= cost_eclipse_duration_s * (1.0 + tolerance) + 1e-9
        )
        if not eclipse_duration_sufficient:
            warnings.append(
                "runtime maximum shadow duration exceeds the battery sizing assumption"
            )
    sunlight_contract_declared = bool(
        expected_sunlit_fraction is not None or cost_eclipse_duration_s is not None
    )
    sunlight_audit_required = bool(
        runtime_energy_ledger_expected and sunlight_contract_declared
    )
    sunlight_audit_valid = bool(
        not sunlight_audit_required
        or (
            sunlight_summary_evaluated
            and (
                expected_sunlit_fraction is None
                or (
                    0.0 <= expected_sunlit_fraction <= 1.0
                    and actual_sunlit_fraction is not None
                    and 0.0 <= actual_sunlit_fraction <= 1.0
                )
            )
            and (
                cost_eclipse_duration_s is None
                or runtime_maximum_shadow_s is not None
            )
        )
    )
    if sunlight_audit_required and not sunlight_audit_valid:
        warnings.append(
            "runtime sunlight evidence is missing or invalid for the cost-model energy contract"
        )

    difference_fraction = None
    numeric_match = False
    actual_dynamic_j = actual["runtime_dynamic_energy_j"]
    if planned_dynamic_j is not None and planned_dynamic_j >= 0 and ledger_available:
        if planned_dynamic_j == 0:
            numeric_match = abs(actual_dynamic_j) <= 1e-12
            difference_fraction = 0.0 if numeric_match else None
            if not numeric_match:
                warnings.append("runtime consumed dynamic energy while planned dynamic energy is zero")
        else:
            difference_fraction = abs(actual_dynamic_j - planned_dynamic_j) / planned_dynamic_j
            numeric_match = difference_fraction <= tolerance

    per_satellite_actual = _mapping(actual.get("per_satellite_dynamic_energy_j"))
    per_satellite_overruns: dict[str, dict[str, float]] = {}
    if planned_per_satellite_buffer_j is not None:
        for satellite_id, raw_value in per_satellite_actual.items():
            actual_value = _finite_nonnegative_or_none(raw_value)
            if actual_value is None:
                continue
            if actual_value > planned_per_satellite_buffer_j * (1.0 + tolerance) + 1e-12:
                per_satellite_overruns[str(satellite_id)] = {
                    "actual_dynamic_energy_j": actual_value,
                    "planned_buffer_energy_j": planned_per_satellite_buffer_j,
                }
    actual["per_satellite_dynamic_buffer_energy_j"] = planned_per_satellite_buffer_j
    actual["per_satellite_overruns"] = per_satellite_overruns
    if per_satellite_overruns:
        warnings.append(
            "one or more satellites exceeded the planned per-satellite workload share; "
            "resizing is required only if the SOC ledger reports a physical energy violation"
        )
    runtime_energy_violation_count = _first_finite(
        _nested(runtime, "runtime_metrics", "energy_violation_count"),
        _nested(runtime, "slo", "energy_violation_count"),
    )
    runtime_energy_violation = bool(
        (runtime_energy_violation_count or 0.0) > 0.0
        or actual["unserved_request_energy_j"] > 0.0
        or actual["request_energy_violation_event_count"] > 0
    )
    if runtime_energy_violation:
        warnings.append("runtime reported an energy violation or unserved request energy")

    evaluated = bool(
        planned_evaluated
        and runtime_evaluated
        and planned_dynamic_j is not None
        and planned_dynamic_j >= 0
        and ledger_available
        and not actual["invalid_ledger_entry_count"]
        and not actual["missing_required_ledger_record_count"]
        and not actual["missing_runtime_request_record_count"]
        and not actual["runtime_request_record_mismatch"]
        and sunlight_audit_valid
    )
    consistent = bool(
        evaluated
        and numeric_match
        and coefficients_match is True
        and sunlight_fraction_match is not False
        and eclipse_duration_sufficient is not False
        and not per_satellite_overruns
        and not runtime_energy_violation
        and target_met is not False
    )
    if evaluated and not numeric_match:
        warnings.append("runtime dynamic energy differs from the offered-workload sizing plan")
    underplanned = bool(
        evaluated
        and planned_dynamic_j is not None
        and actual_dynamic_j > planned_dynamic_j * (1.0 + tolerance) + 1e-12
    )
    sunlight_underplanned = bool(
        expected_sunlit_fraction is not None
        and actual_sunlit_fraction is not None
        and actual_sunlit_fraction + tolerance + 1e-12 < expected_sunlit_fraction
    )
    eclipse_underplanned = eclipse_duration_sufficient is False
    requires_resizing = bool(
        underplanned
        or (ledger_available and coefficients_match is False)
        or actual["invalid_ledger_entry_count"] > 0
        or actual["missing_required_ledger_record_count"] > 0
        or actual["missing_runtime_request_record_count"] > 0
        or actual["runtime_request_record_mismatch"]
        or (sunlight_audit_required and not sunlight_audit_valid)
        or sunlight_underplanned
        or eclipse_underplanned
        or runtime_energy_violation
    )
    if requires_resizing:
        warnings.append(
            "runtime energy, ledger, or sunlight audit requires construction resizing"
        )

    return EnergyAlignmentResult(
        evaluated=evaluated,
        planned_energy={
            "planned_compute_energy_j": planned_compute_j,
            "planned_link_energy_j": planned_link_j,
            "planned_dynamic_energy_j": planned_dynamic_j,
            "offered_request_count": offered_request_count,
            "per_satellite_dynamic_energy_j": planned_per_satellite_dynamic_j,
            "maximum_request_dynamic_energy_j": planned_maximum_request_j,
            "per_satellite_dynamic_buffer_energy_j": planned_per_satellite_buffer_j,
            "coefficients": planned_coefficients,
            "source": _first_present(
                planned.get("demand_source"),
                planned.get("source"),
                _nested(planned, "provenance", "demand_source"),
                "offered_workload",
            ),
        },
        runtime_energy=actual,
        difference_fraction=difference_fraction,
        tolerance_fraction=tolerance,
        target_served_ratio=target_served_ratio,
        runtime_completion_ratio=runtime_completion_ratio,
        target_served_ratio_met=target_met,
        coefficients_match=coefficients_match,
        sunlight_fraction_match=sunlight_fraction_match,
        eclipse_duration_sufficient=eclipse_duration_sufficient,
        sunlight_contract={
            "cost_sizing_sunlit_fraction": expected_sunlit_fraction,
            "runtime_actual_sunlit_fraction": actual_sunlit_fraction,
            "absolute_difference": sunlight_difference,
            "tolerance": tolerance,
            "cost_sizing_eclipse_duration_s": cost_eclipse_duration_s,
            "runtime_maximum_shadow_window_s": runtime_maximum_shadow_s,
            "runtime_summary": sunlight_summary,
            "audit_required": sunlight_audit_required,
            "audit_valid": sunlight_audit_valid,
        },
        consistent=consistent,
        requires_resizing=requires_resizing,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance={
            "sizing_input": "offered_workload",
            "runtime_records_used_for_cost_sizing": False,
            "runtime_records_role": "post_runtime_audit_only",
            "planned_coefficients": planned_coefficients,
            "runtime_coefficients": runtime_coefficients,
            "envelope_coefficients_match": envelope_coefficients_match,
            "runtime_ledger_coefficient_audit": ledger_coefficient_audit,
            "runtime_energy_ledger_expected": runtime_energy_ledger_expected,
            "runtime_sunlight_audit_required": sunlight_audit_required,
            "runtime_sunlight_audit_valid": sunlight_audit_valid,
            "runtime_energy_violation_count": runtime_energy_violation_count,
            "per_satellite_overruns": per_satellite_overruns,
        },
    )


def attach_energy_alignment(
    runtime_performance: Any,
    planned_energy_demand: Any,
    *,
    runtime_energy_envelope: Any = None,
    runtime_sunlight_summary: Any = None,
    tolerance_fraction: float = DEFAULT_ALIGNMENT_TOLERANCE_FRACTION,
) -> tuple[Any, EnergyAlignmentResult]:
    """Return runtime output with a post-runtime alignment payload attached."""

    alignment = evaluate_energy_alignment(
        planned_energy_demand,
        runtime_performance,
        runtime_energy_envelope=runtime_energy_envelope,
        runtime_sunlight_summary=runtime_sunlight_summary,
        tolerance_fraction=tolerance_fraction,
    )
    existing_details = (
        runtime_performance.get("details")
        if isinstance(runtime_performance, Mapping)
        else getattr(runtime_performance, "details", None)
    )
    details = {
        **_mapping(existing_details),
        "planned_energy_demand": _mapping(planned_energy_demand),
        "energy_alignment": alignment.to_dict(),
    }
    if is_dataclass(runtime_performance):
        return replace(runtime_performance, details=details), alignment
    if isinstance(runtime_performance, dict):
        payload = dict(runtime_performance)
        payload["details"] = details
        return payload, alignment
    return runtime_performance, alignment


def summarize_runtime_sunlight(energy_windows: Any) -> dict[str, Any]:
    """Summarize the actual sunlit duration represented by runtime windows."""

    total_duration_s = 0.0
    sunlit_duration_s = 0.0
    maximum_shadow_window_s = 0.0
    satellites: set[str] = set()
    invalid_window_count = 0
    for raw in tuple(energy_windows or ()):
        if not isinstance(raw, Mapping):
            invalid_window_count += 1
            continue
        start_s = _finite_nonnegative_or_none(raw.get("start_s"))
        end_s = _finite_nonnegative_or_none(raw.get("end_s"))
        if start_s is None or end_s is None or end_s < start_s:
            invalid_window_count += 1
            continue
        duration_s = end_s - start_s
        satellites.add(str(raw.get("satellite_id") or "<unknown>"))
        total_duration_s += duration_s
        if bool(raw.get("sunlit", False)):
            sunlit_duration_s += duration_s
        else:
            maximum_shadow_window_s = max(maximum_shadow_window_s, duration_s)
    evaluated = total_duration_s > 0 and invalid_window_count == 0
    return {
        "evaluated": evaluated,
        "actual_sunlit_fraction": (
            sunlit_duration_s / total_duration_s if total_duration_s > 0 else None
        ),
        "total_window_duration_s": total_duration_s,
        "sunlit_window_duration_s": sunlit_duration_s,
        "maximum_shadow_window_s": maximum_shadow_window_s,
        "satellite_count": len(satellites),
        "invalid_window_count": invalid_window_count,
    }


def _aggregate_request_event_energy(
    records: tuple[dict[str, Any], ...],
    *,
    require_request_event_ledger: bool,
) -> dict[str, Any]:
    seen: set[tuple[str, str]] = set()
    compute_j = 0.0
    link_j = 0.0
    per_satellite_dynamic_j: dict[str, float] = {}
    unserved_request_energy_j = 0.0
    request_energy_violation_event_count = 0
    missing_required_ledger_record_count = 0
    invalid = 0
    event_count = 0
    for record_index, record in enumerate(records):
        ledger = _request_energy_ledger(record)
        request_event_rows = tuple(
            row
            for row in ledger
            if isinstance(row, dict) and row.get("energy_kind") == "request_event"
        )
        if (
            require_request_event_ledger
            and _record_requires_request_energy_ledger(record)
            and not request_event_rows
        ):
            missing_required_ledger_record_count += 1
        for entry_index, raw in enumerate(ledger):
            if not isinstance(raw, dict) or raw.get("energy_kind") != "request_event":
                continue
            satellite_id = str(raw.get("satellite_id") or "").strip()
            event_id = str(raw.get("event_id") or "").strip()
            if not satellite_id or satellite_id == "<unknown>" or not event_id:
                invalid += 1
                continue
            identity = (satellite_id, event_id)
            if identity in seen:
                continue
            seen.add(identity)
            compute_value = _finite_nonnegative_or_none(raw.get("compute_bus_j", 0.0))
            link_value = _finite_nonnegative_or_none(raw.get("link_bus_j", 0.0))
            if compute_value is None or link_value is None:
                invalid += 1
                continue
            compute_j += compute_value
            link_j += link_value
            per_satellite_dynamic_j[satellite_id] = (
                per_satellite_dynamic_j.get(satellite_id, 0.0)
                + compute_value
                + link_value
            )
            unserved_value = _finite_nonnegative_or_none(
                raw.get("unserved_energy_j", 0.0)
            )
            if unserved_value is None:
                invalid += 1
            else:
                unserved_request_energy_j += unserved_value
            soc_after = _finite_nonnegative_or_none(raw.get("soc_after"))
            soc_min = _finite_nonnegative_or_none(raw.get("soc_min"))
            if (
                (unserved_value or 0.0) > 0.0
                or (
                    soc_after is not None
                    and soc_min is not None
                    and soc_after + 1e-12 < soc_min
                )
            ):
                request_energy_violation_event_count += 1
            event_count += 1
    completed_count = sum(1 for record in records if bool(record.get("completed", False)))
    return {
        "attempted_request_count": len(records),
        "completed_request_count": completed_count,
        "request_event_count": event_count,
        "deduplicated_request_event_count": len(seen),
        "invalid_ledger_entry_count": invalid,
        "missing_required_ledger_record_count": missing_required_ledger_record_count,
        "runtime_compute_energy_j": compute_j,
        "runtime_link_energy_j": link_j,
        "runtime_dynamic_energy_j": compute_j + link_j,
        "per_satellite_dynamic_energy_j": dict(sorted(per_satellite_dynamic_j.items())),
        "unserved_request_energy_j": unserved_request_energy_j,
        "request_energy_violation_event_count": request_energy_violation_event_count,
    }


def _runtime_energy_ledger_expected(runtime: dict[str, Any]) -> bool:
    """Return false only for an explicitly compute-only runtime contract."""

    scope = str(
        _first_present(
            runtime.get("scope"),
            _nested(runtime, "runtime_metrics", "scope"),
            "",
        )
        or ""
    )
    trace_model = str(
        _first_present(
            runtime.get("trace_model"),
            _nested(runtime, "runtime_metrics", "trace_model"),
            "",
        )
        or ""
    )
    return not (
        scope == "idealized_compute_only"
        or trace_model == "ideal_full_visibility"
    )


def _record_requires_request_energy_ledger(record: dict[str, Any]) -> bool:
    """Only satellite-completed work must carry a satellite energy ledger."""

    if not bool(record.get("completed", False)):
        return False
    action_type = str(record.get("action_type") or "")
    if bool(record.get("fallback_used", False)) or action_type.startswith("ground_"):
        return False
    return True


def _audit_runtime_ledger_coefficients(
    records: tuple[dict[str, Any], ...],
    planned_coefficients: dict[str, Any],
    *,
    allow_empty: bool,
) -> dict[str, Any]:
    """Verify every runtime debit equals base coefficient x policy scale x work."""

    planned_compute = _finite_nonnegative_or_none(
        planned_coefficients.get("compute_energy_j_per_token")
    )
    planned_link = _finite_nonnegative_or_none(
        planned_coefficients.get("link_energy_j_per_byte")
    )
    checked = 0
    invalid = 0
    formula_mismatch = 0
    base_mismatch = 0
    seen: set[tuple[str, str]] = set()
    for record_index, record in enumerate(records):
        ledger = _request_energy_ledger(record)
        for entry_index, row in enumerate(ledger):
            if not isinstance(row, dict) or row.get("energy_kind") != "request_event":
                continue
            identity = (
                str(row.get("satellite_id") or "<unknown>"),
                str(row.get("event_id") or f"record-{record_index}-entry-{entry_index}"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            values = {
                key: _finite_nonnegative_or_none(row.get(key))
                for key in (
                    "compute_tokens",
                    "link_bytes_accounted",
                    "compute_bus_j",
                    "link_bus_j",
                    "compute_energy_j_per_token",
                    "link_energy_j_per_byte",
                    "base_compute_energy_j_per_token",
                    "base_link_energy_j_per_byte",
                    "runtime_compute_energy_scale",
                    "runtime_link_energy_scale",
                )
            }
            checked += 1
            if any(value is None for value in values.values()) or planned_compute is None or planned_link is None:
                invalid += 1
                continue
            if not math.isclose(
                values["base_compute_energy_j_per_token"],
                planned_compute,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ) or not math.isclose(
                values["base_link_energy_j_per_byte"],
                planned_link,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                base_mismatch += 1
            expected_compute_coefficient = (
                values["base_compute_energy_j_per_token"]
                * values["runtime_compute_energy_scale"]
            )
            expected_link_coefficient = (
                values["base_link_energy_j_per_byte"]
                * values["runtime_link_energy_scale"]
            )
            expected_compute_j = values["compute_tokens"] * values["compute_energy_j_per_token"]
            expected_link_j = values["link_bytes_accounted"] * values["link_energy_j_per_byte"]
            checks = (
                math.isclose(
                    values["compute_energy_j_per_token"],
                    expected_compute_coefficient,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ),
                math.isclose(
                    values["link_energy_j_per_byte"],
                    expected_link_coefficient,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ),
                math.isclose(
                    values["compute_bus_j"],
                    expected_compute_j,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                math.isclose(
                    values["link_bus_j"],
                    expected_link_j,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
            )
            if not all(checks):
                formula_mismatch += 1
    coefficients_match = (
        True
        if checked == 0 and allow_empty and planned_compute is not None and planned_link is not None
        else checked > 0 and invalid == 0 and formula_mismatch == 0 and base_mismatch == 0
    )
    return {
        "checked_request_event_count": checked,
        "invalid_coefficient_entry_count": invalid,
        "formula_mismatch_count": formula_mismatch,
        "base_coefficient_mismatch_count": base_mismatch,
        "coefficients_match": coefficients_match,
    }


def _request_energy_ledger(record: dict[str, Any]) -> tuple[Any, ...]:
    """Read the one supported ledger from either serialized record shape."""

    ledger = record.get("energy_ledger")
    if not isinstance(ledger, (list, tuple)):
        ledger = _nested(record, "provenance", "energy_ledger")
    return tuple(ledger) if isinstance(ledger, (list, tuple)) else ()


def _runtime_request_records(runtime_performance: Any, runtime: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = getattr(runtime_performance, "request_records", None)
    if raw is None:
        raw = runtime.get("request_records")
    if not raw:
        raw = _nested(runtime, "details", "request_records")
    return tuple(dict(item) for item in (raw or ()) if isinstance(item, dict))


def _planned_coefficients(planned: dict[str, Any]) -> dict[str, Any]:
    for value in (
        planned.get("coefficients"),
        planned.get("energy_coefficients"),
        planned.get("energy_intensity"),
        _nested(planned, "provenance", "coefficients"),
    ):
        if isinstance(value, dict):
            return dict(value)
    keys = (
        "compute_energy_j_per_token",
        "prefill_energy_j_per_token",
        "decode_energy_j_per_token",
        "link_energy_j_per_byte",
        "sgl_tx_energy_j_per_byte",
        "sgl_rx_energy_j_per_byte",
        "isl_tx_energy_j_per_byte",
        "isl_rx_energy_j_per_byte",
    )
    return {key: planned[key] for key in keys if key in planned}


def _runtime_coefficients(runtime_energy_envelope: Any) -> dict[str, Any]:
    envelope = _mapping(runtime_energy_envelope)
    options = _mapping(envelope.get("trace_options"))
    keys = (
        "compute_energy_j_per_token",
        "prefill_energy_j_per_token",
        "decode_energy_j_per_token",
        "link_energy_j_per_byte",
        "sgl_tx_energy_j_per_byte",
        "sgl_rx_energy_j_per_byte",
        "isl_tx_energy_j_per_byte",
        "isl_rx_energy_j_per_byte",
    )
    return {key: options[key] for key in keys if key in options}


def _coefficient_match(planned: dict[str, Any], runtime: dict[str, Any]) -> bool | None:
    shared = set(planned) & set(runtime)
    if not shared:
        return None
    for key in shared:
        left = _finite_nonnegative_or_none(planned[key])
        right = _finite_nonnegative_or_none(runtime[key])
        if left is None or right is None or not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15):
            return False
    return True


def _target_served_ratio(planned: dict[str, Any]) -> float | None:
    value = _first_finite(
        planned.get("target_served_ratio"),
        planned.get("sizing_target_served_ratio"),
        _nested(planned, "sizing", "target_served_ratio"),
        _nested(planned, "provenance", "target_served_ratio"),
    )
    return value if value is not None and 0 <= value <= 1 else None


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, dict) else {}
    converted = to_jsonable(value)
    return dict(converted) if isinstance(converted, dict) else {}


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_finite(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_nonnegative_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _finite_nonnegative_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _nonnegative_finite(value: Any, name: str) -> float:
    parsed = _finite_nonnegative_or_none(value)
    if parsed is None:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


__all__ = [
    "DEFAULT_ALIGNMENT_TOLERANCE_FRACTION",
    "EnergyAlignmentResult",
    "attach_energy_alignment",
    "evaluate_energy_alignment",
    "summarize_runtime_sunlight",
]
