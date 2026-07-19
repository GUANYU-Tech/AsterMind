"""Top-level cost-SLO frontier contract for OrbitServe DSE studies."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping

from ..common import to_jsonable


CORE_PARETO_FIELDS = (
    "annualized_cost_usd_per_year",
    "served_ratio",
    "e2e_p95_s",
)

DOMINANCE_OBJECTIVES = (
    {
        "field": "annualized_cost_usd_per_year",
        "direction": "minimize",
    },
    {"field": "served_ratio", "direction": "maximize"},
    {"field": "e2e_p95_s", "direction": "minimize"},
)


@dataclass(frozen=True)
class DSEFrontierCandidate:
    candidate_id: str
    design_strategy_id: str
    row_index: int
    design_id: str
    strategy_id: str
    policy: str
    source: str
    runtime_evaluated: bool
    eligible: bool
    excluded_reason: str | None
    missing_fields: tuple[str, ...]
    pareto_member: bool
    valid_for_paper_main_results: bool
    runtime_metric_scope: str | None
    calibration: str
    cost_metric: str | None
    cost_usd: float | None
    annualized_cost_usd_per_year: float | None
    cost_total_usd: float | None
    served_ratio: float | None
    e2e_p95_s: float | None
    ttft_p95_s: float | None
    e2e_p99_s: float | None
    ttft_p99_s: float | None
    objective_score: float | None
    objective_score_scope: str | None
    trace_model: str | None
    scope: str | None
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class DSEFrontierResult:
    evaluated: bool = False
    analysis_scope: str = "not_evaluated"
    valid_for_paper_main_results: bool = False
    calibration: str = "not_evaluated"
    candidate_count: int = 0
    eligible_count: int = 0
    excluded_count: int = 0
    frontier_count: int = 0
    dominated_candidate_count: int | None = None
    dominance_objectives: tuple[dict[str, Any], ...] = DOMINANCE_OBJECTIVES
    frontier_rows: tuple[dict[str, Any], ...] = ()
    eligible_candidates: tuple[dict[str, Any], ...] = ()
    excluded_candidates: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def unevaluated_dse_frontier(reason: str = "DSE frontier is not evaluated") -> DSEFrontierResult:
    return DSEFrontierResult(
        provenance={
            "source": "runtime_matrix",
            "status": "not_evaluated",
            "calibrated_paper_main": "reserved_not_implemented",
        },
        warnings=(reason,),
    )


def evaluate_dse_frontier(runtime_matrix: Any) -> DSEFrontierResult:
    """Compute a preliminary top-level DSE cost-SLO frontier from runtime matrix rows."""

    matrix = _mapping_from_value(runtime_matrix)
    rows = tuple(_matrix_rows(runtime_matrix, matrix))
    if not bool(matrix.get("evaluated", False)):
        return DSEFrontierResult(
            candidate_count=len(rows),
            excluded_count=len(rows),
            excluded_candidates=tuple(
                _excluded_payload(row, "runtime_matrix is not evaluated", row_index=index)
                for index, row in enumerate(rows)
            ),
            provenance=_frontier_provenance(status="blocked_runtime_matrix_not_evaluated"),
            warnings=(
                "top-level DSE frontier is not evaluated because runtime_matrix.evaluated=false",
                "calibrated_paper_main frontier is reserved for a future STK/equivalent calibration path",
            ),
        )

    allow_preliminary = bool(matrix.get("allow_preliminary_runtime_metrics", False))
    if not allow_preliminary:
        return DSEFrontierResult(
            candidate_count=len(rows),
            excluded_count=len(rows),
            excluded_candidates=tuple(
                _excluded_payload(
                    row,
                    "preliminary runtime metrics are not allowed in strict mode",
                    row_index=index,
                )
                for index, row in enumerate(rows)
            ),
            provenance=_frontier_provenance(status="blocked_strict_mode"),
            warnings=(
                "strict DSE mode leaves the top-level cost-SLO frontier unevaluated",
                "set runtime_matrix.allow_preliminary_runtime_metrics=true to compute a preliminary_surrogate frontier",
                "calibrated_paper_main frontier is reserved for a future STK/equivalent calibration path",
            ),
        )

    draft_candidates = tuple(
        _candidate_from_row(row, row_index=index, source="runtime_matrix")
        for index, row in enumerate(rows)
    )
    eligible = tuple(candidate for candidate in draft_candidates if candidate.eligible)
    excluded = tuple(candidate for candidate in draft_candidates if not candidate.eligible)
    if not eligible:
        return DSEFrontierResult(
            analysis_scope="preliminary_surrogate",
            calibration="not_stk_calibrated",
            candidate_count=len(draft_candidates),
            excluded_count=len(excluded),
            excluded_candidates=tuple(candidate.to_dict() for candidate in excluded),
            provenance=_frontier_provenance(status="blocked_no_eligible_candidates"),
            warnings=(
                "top-level DSE frontier is not evaluated because no runtime_matrix rows have valid complete cost/SLO/tail-latency fields",
                f"required Pareto fields: {', '.join(CORE_PARETO_FIELDS)}",
                "calibrated_paper_main frontier is reserved for a future STK/equivalent calibration path",
            ),
        )

    pareto_ids = _pareto_candidate_ids(eligible)
    candidates = tuple(
        _with_pareto_member(candidate, candidate.candidate_id in pareto_ids)
        for candidate in draft_candidates
    )
    eligible_candidates = tuple(candidate for candidate in candidates if candidate.eligible)
    frontier_candidates = tuple(candidate for candidate in eligible_candidates if candidate.pareto_member)
    frontier_rows = tuple(
        candidate.to_dict()
        for candidate in sorted(frontier_candidates, key=_frontier_sort_key)
    )
    warnings = [
        "top-level DSE frontier is preliminary_surrogate, not STK-calibrated, and not valid for paper main results",
        "frontier dominance uses annualized_cost_usd_per_year, served_ratio, and e2e_p95_s instead of weighted objective_score alone",
        "cost_total_usd is preserved in candidate payloads but is not used for preliminary frontier dominance",
        "calibrated_paper_main frontier is reserved for a future STK/equivalent calibration path",
    ]
    if excluded:
        warnings.append(
            "some runtime_matrix rows were excluded from the top-level frontier because they failed gates or lacked core Pareto fields"
        )
    return DSEFrontierResult(
        evaluated=True,
        analysis_scope="preliminary_surrogate",
        valid_for_paper_main_results=False,
        calibration="not_stk_calibrated",
        candidate_count=len(candidates),
        eligible_count=len(eligible_candidates),
        excluded_count=len(excluded),
        frontier_count=len(frontier_rows),
        dominated_candidate_count=len(eligible_candidates) - len(frontier_rows),
        frontier_rows=frontier_rows,
        eligible_candidates=tuple(candidate.to_dict() for candidate in eligible_candidates),
        excluded_candidates=tuple(candidate.to_dict() for candidate in candidates if not candidate.eligible),
        provenance=_frontier_provenance(status="evaluated_preliminary_surrogate"),
        warnings=tuple(warnings),
    )


def evaluate_static_slo_frontier(rows: Any) -> DSEFrontierResult:
    """Compute a design-only static workload-SLO frontier from DSE rows."""

    row_payloads = tuple(_static_row_payload(row) for row in rows)
    if not row_payloads:
        return unevaluated_dse_frontier("static workload-SLO frontier is not evaluated because there are no DSE rows")
    draft_candidates = tuple(
        _candidate_from_row(row, row_index=index, source="static_slo_proxy")
        for index, row in enumerate(row_payloads)
    )
    eligible = tuple(candidate for candidate in draft_candidates if candidate.eligible)
    excluded = tuple(candidate for candidate in draft_candidates if not candidate.eligible)
    if not eligible:
        return DSEFrontierResult(
            analysis_scope="design_only_surrogate",
            calibration="not_stk_calibrated",
            candidate_count=len(draft_candidates),
            excluded_count=len(excluded),
            excluded_candidates=tuple(candidate.to_dict() for candidate in excluded),
            provenance=_frontier_provenance(
                source="static_slo_proxy",
                status="blocked_no_eligible_static_slo_candidates",
            ),
            warnings=(
                "static workload-SLO frontier is not evaluated because no DSE rows have complete static cost/SLO/tail-latency fields",
                f"required Pareto fields: {', '.join(CORE_PARETO_FIELDS)}",
                "calibrated_paper_main frontier is reserved for a future STK/equivalent calibration path",
            ),
        )

    pareto_ids = _pareto_candidate_ids(eligible)
    candidates = tuple(
        _with_pareto_member(candidate, candidate.candidate_id in pareto_ids)
        for candidate in draft_candidates
    )
    eligible_candidates = tuple(candidate for candidate in candidates if candidate.eligible)
    frontier_candidates = tuple(candidate for candidate in eligible_candidates if candidate.pareto_member)
    frontier_rows = tuple(
        candidate.to_dict()
        for candidate in sorted(frontier_candidates, key=_frontier_sort_key)
    )
    warnings = [
        "static workload-SLO frontier is design_only_surrogate, not STK-calibrated, and not valid for paper main results",
        "frontier dominance uses annualized_cost_usd_per_year, served_ratio, and e2e_p95_s",
        "static proxy uses request arrivals and timebin demand but does not replace dynamic scheduler/runtime evaluation",
        "calibrated_paper_main frontier is reserved for a future STK/equivalent calibration path",
    ]
    if excluded:
        warnings.append(
            "some DSE rows were excluded from the static frontier because they lacked complete static proxy metrics"
        )
    return DSEFrontierResult(
        evaluated=True,
        analysis_scope="design_only_surrogate",
        valid_for_paper_main_results=False,
        calibration="not_stk_calibrated",
        candidate_count=len(candidates),
        eligible_count=len(eligible_candidates),
        excluded_count=len(excluded),
        frontier_count=len(frontier_rows),
        dominated_candidate_count=len(eligible_candidates) - len(frontier_rows),
        frontier_rows=frontier_rows,
        eligible_candidates=tuple(candidate.to_dict() for candidate in eligible_candidates),
        excluded_candidates=tuple(candidate.to_dict() for candidate in candidates if not candidate.eligible),
        provenance=_frontier_provenance(source="static_slo_proxy", status="evaluated_design_only_surrogate"),
        warnings=tuple(warnings),
    )


def _candidate_from_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    source: str,
) -> DSEFrontierCandidate:
    design_id = str(row.get("design_id") or "")
    strategy_id = str(row.get("strategy_id") or "")
    design_strategy_id = f"{design_id}::{strategy_id}"
    candidate_id = f"{design_strategy_id}::row_{row_index}"
    annualized_cost = _optional_float(row.get("annualized_cost_usd_per_year"))
    total_cost = _optional_float(_first_present(row.get("cost_total_usd"), row.get("total_cost_usd")))
    annualized_cost_valid = annualized_cost is not None and annualized_cost >= 0.0
    if annualized_cost_valid:
        cost_metric = "annualized_cost_usd_per_year"
        cost_usd = annualized_cost
    else:
        cost_metric = None
        cost_usd = None

    served_ratio = _optional_float(row.get("served_ratio"))
    e2e_p95_s = _optional_float(row.get("e2e_p95_s"))
    missing_fields = []
    if not annualized_cost_valid:
        missing_fields.append("annualized_cost_usd_per_year")
    if served_ratio is None or not 0.0 <= served_ratio <= 1.0:
        missing_fields.append("served_ratio")
    if e2e_p95_s is None or e2e_p95_s < 0.0:
        missing_fields.append("e2e_p95_s")

    runtime_evaluated = bool(row.get("runtime_evaluated", False))
    valid_for_paper = bool(row.get("valid_for_paper_main_results", False))
    row_provenance = _mapping_from_value(row.get("provenance"))
    runtime_details = _mapping_from_value(
        _mapping_from_value(row.get("runtime_result")).get("details")
    )
    energy_alignment = _mapping_from_value(
        _first_present(row.get("energy_alignment"), runtime_details.get("energy_alignment"))
    )
    cost_feasible = _first_present(row.get("cost_feasible"), row_provenance.get("cost_feasible"))
    alignment_runtime_energy = _mapping_from_value(energy_alignment.get("runtime_energy"))
    energy_requires_resizing = bool(
        energy_alignment.get("requires_resizing", False)
        or (_optional_float(alignment_runtime_energy.get("invalid_ledger_entry_count")) or 0.0)
        > 0.0
    )
    excluded_reason = None
    if not runtime_evaluated:
        excluded_reason = (
            "static_slo_evaluated is false"
            if source == "static_slo_proxy"
            else "runtime_evaluated is false"
        )
    elif valid_for_paper:
        excluded_reason = "calibrated_paper_main frontier is reserved; current evaluator only publishes preliminary_surrogate frontier"
    elif cost_feasible is False:
        excluded_reason = "cost model marked the design infeasible"
    elif energy_requires_resizing:
        excluded_reason = "energy alignment requires you to resize the design before frontier selection"
    elif missing_fields:
        excluded_reason = f"missing/invalid core Pareto fields: {', '.join(missing_fields)}"

    provenance = {
        "source": "runtime_matrix.matrix_rows" if source == "runtime_matrix" else "dse.rows.static_slo_proxy",
        "runtime_matrix_provenance": row_provenance,
        "cost_feasible": cost_feasible,
        "energy_alignment": energy_alignment,
    }
    return DSEFrontierCandidate(
        candidate_id=candidate_id,
        design_strategy_id=design_strategy_id,
        row_index=row_index,
        design_id=design_id,
        strategy_id=strategy_id,
        policy=str(row.get("policy") or ""),
        source=source,
        runtime_evaluated=runtime_evaluated,
        eligible=excluded_reason is None,
        excluded_reason=excluded_reason,
        missing_fields=tuple(missing_fields),
        pareto_member=False,
        valid_for_paper_main_results=valid_for_paper,
        runtime_metric_scope=_optional_str(row.get("objective_score_scope")),
        calibration="not_stk_calibrated",
        cost_metric=cost_metric,
        cost_usd=cost_usd,
        annualized_cost_usd_per_year=annualized_cost,
        cost_total_usd=total_cost,
        served_ratio=served_ratio,
        e2e_p95_s=e2e_p95_s,
        ttft_p95_s=_optional_float(row.get("ttft_p95_s")),
        e2e_p99_s=_optional_float(row.get("e2e_p99_s")),
        ttft_p99_s=_optional_float(row.get("ttft_p99_s")),
        objective_score=_optional_float(row.get("objective_score")),
        objective_score_scope=_optional_str(row.get("objective_score_scope")),
        trace_model=_optional_str(row.get("trace_model")),
        scope=_optional_str(row.get("scope")),
        warnings=_string_tuple(row.get("warnings")),
        provenance=provenance,
    )


def _static_row_payload(row: Any) -> dict[str, Any]:
    details = _static_row_performance_details(row)
    proxy = _mapping_from_value(details.get("static_slo_proxy"))
    proxy_provenance = _mapping_from_value(proxy.get("provenance"))
    proxy_access = _mapping_from_value(proxy.get("access_summary"))
    return {
        "design_id": _row_field(row, "design_id"),
        "strategy_id": "static_slo_proxy",
        "policy": "static_slo_proxy",
        "runtime_evaluated": bool(_row_section_field(row, "slo", "evaluated") or False),
        "valid_for_paper_main_results": False,
        "cost_feasible": _row_section_field(row, "cost", "single_satellite") is not None
        and bool(
            _mapping_from_value(_row_section_field(row, "cost", "single_satellite")).get(
                "feasible", False
            )
        ),
        "energy_alignment": _mapping_from_value(details.get("energy_alignment")),
        "annualized_cost_usd_per_year": _row_section_field(row, "cost", "annualized_cost_usd_per_year"),
        "cost_total_usd": _row_section_field(row, "cost", "total_cost_usd"),
        "served_ratio": _row_section_field(row, "slo", "served_ratio"),
        "rejected_ratio": _row_section_field(row, "slo", "rejected_ratio"),
        "timeout_ratio": _row_section_field(row, "slo", "timeout_ratio"),
        "ttft_p95_s": _row_section_field(row, "performance", "ttft_p95_s"),
        "ttft_p99_s": _row_section_field(row, "performance", "ttft_p99_s"),
        "e2e_p95_s": _row_section_field(row, "performance", "e2e_p95_s"),
        "e2e_p99_s": _row_section_field(row, "performance", "e2e_p99_s"),
        "trace_model": proxy_access.get("trace_model"),
        "scope": proxy_provenance.get("analysis_scope") or "design_only_surrogate",
        "objective_score": None,
        "objective_score_scope": (
            "design_only_surrogate" if bool(_row_section_field(row, "slo", "evaluated") or False) else None
        ),
        "warnings": tuple(
            dict.fromkeys(
                (
                    *_string_tuple(_row_field(row, "warnings")),
                    *_string_tuple(_row_section_field(row, "performance", "warnings")),
                    *_string_tuple(_row_section_field(row, "slo", "warnings")),
                )
            )
        ),
        "provenance": {
            "design_id": _row_field(row, "design_id"),
            "source": "DSEDesignRow.slo/performance",
            "static_slo_proxy": proxy.get("provenance"),
        },
    }


def _row_field(row: Any, field_name: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(field_name)
    return getattr(row, field_name, None)


def _row_section_field(row: Any, section_name: str, field_name: str) -> Any:
    section = _row_field(row, section_name)
    if isinstance(section, Mapping):
        return section.get(field_name)
    return getattr(section, field_name, None)


def _static_row_performance_details(row: Any) -> dict[str, Any]:
    details = _row_section_field(row, "performance", "details")
    return details if isinstance(details, dict) else _mapping_from_value(details)


def _with_pareto_member(candidate: DSEFrontierCandidate, pareto_member: bool) -> DSEFrontierCandidate:
    return replace(candidate, pareto_member=pareto_member)


def _pareto_candidate_ids(candidates: tuple[DSEFrontierCandidate, ...]) -> set[str]:
    pareto: set[str] = set()
    for candidate in candidates:
        if not any(_dominates(other, candidate) for other in candidates if other is not candidate):
            pareto.add(candidate.candidate_id)
    return pareto


def _dominates(left: DSEFrontierCandidate, right: DSEFrontierCandidate) -> bool:
    if left.cost_usd is None or right.cost_usd is None:
        return False
    if left.served_ratio is None or right.served_ratio is None:
        return False
    if left.e2e_p95_s is None or right.e2e_p95_s is None:
        return False
    no_worse = (
        left.cost_usd <= right.cost_usd
        and left.served_ratio >= right.served_ratio
        and left.e2e_p95_s <= right.e2e_p95_s
    )
    strictly_better = (
        left.cost_usd < right.cost_usd
        or left.served_ratio > right.served_ratio
        or left.e2e_p95_s < right.e2e_p95_s
    )
    return no_worse and strictly_better


def _frontier_sort_key(candidate: DSEFrontierCandidate) -> tuple[float, float, float, str, str, int]:
    return (
        float(candidate.cost_usd if candidate.cost_usd is not None else float("inf")),
        -float(candidate.served_ratio if candidate.served_ratio is not None else float("-inf")),
        float(candidate.e2e_p95_s if candidate.e2e_p95_s is not None else float("inf")),
        candidate.design_id,
        candidate.strategy_id,
        candidate.row_index,
    )


def _frontier_provenance(*, status: str, source: str = "runtime_matrix") -> dict[str, Any]:
    analysis_scope = "design_only_surrogate" if source == "static_slo_proxy" else "preliminary_surrogate"
    return {
        "source": source,
        "status": status,
        "analysis_scope": analysis_scope,
        "calibration": "not_stk_calibrated",
        "valid_for_paper_main_results": False,
        "dominance": {
            "algorithm": "non_dominated_selection",
            "cost_basis": "annualized_cost_usd_per_year",
            "objectives": DOMINANCE_OBJECTIVES,
        },
        "calibrated_paper_main": "reserved_not_implemented",
    }


def _excluded_payload(row: Mapping[str, Any], reason: str, *, row_index: int) -> dict[str, Any]:
    design_id = str(row.get("design_id") or "")
    strategy_id = str(row.get("strategy_id") or "")
    design_strategy_id = f"{design_id}::{strategy_id}"
    return {
        "candidate_id": f"{design_strategy_id}::row_{row_index}",
        "design_strategy_id": design_strategy_id,
        "row_index": row_index,
        "design_id": design_id,
        "strategy_id": strategy_id,
        "policy": row.get("policy"),
        "eligible": False,
        "excluded_reason": reason,
        "valid_for_paper_main_results": bool(row.get("valid_for_paper_main_results", False)),
    }


def _matrix_rows(runtime_matrix: Any, matrix: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_rows = getattr(runtime_matrix, "matrix_rows", None)
    if raw_rows is None:
        raw_rows = matrix.get("matrix_rows", ())
    return tuple(_mapping_from_value(row) for row in (raw_rows or ()))


def _mapping_from_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(to_jsonable(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = to_jsonable(value.to_dict())
        return converted if isinstance(converted, dict) else {}
    converted = to_jsonable(value)
    return converted if isinstance(converted, dict) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


__all__ = [
    "DSEFrontierCandidate",
    "DSEFrontierResult",
    "evaluate_dse_frontier",
    "evaluate_static_slo_frontier",
    "unevaluated_dse_frontier",
]
