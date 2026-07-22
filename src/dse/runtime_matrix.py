"""Runtime scheduler matrix contracts and evaluator for DSE studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ..common import to_jsonable
from ..common import DSEMetrics
from ..runtime import RuntimeStrategySpec
from ..experiments.runtime_dispatch import evaluate_runtime_performance
from ..runtime import build_runtime_deployment_catalog
from .runtime_optimization import RuntimeOptimizationObjective, score_runtime_evaluator_result


@dataclass(frozen=True)
class DSEDesignContext:
    design_id: str
    design_row: Any
    runtime_scenario: Any
    current_scheduler: dict[str, Any]
    cost_result: Any


@dataclass(frozen=True)
class DSERuntimeMatrixRow:
    design_id: str
    strategy_id: str
    policy: str
    static_screen_selected: bool
    static_candidate_selected: bool
    static_screen_reason: str | None
    static_rank: int | None
    static_score: float | None
    static_frontier_member: bool
    static_served_ratio: float | None
    static_e2e_p95_s: float | None
    static_ttft_p95_s: float | None
    static_annualized_cost_usd_per_year: float | None
    placement_scope_strategy_id: str | None
    pd_placement_scope: str | None
    pd_pairing_rule: str | None
    pd_scope_candidate_count: int | None
    pd_pair_candidate_count: int | None
    serving_satellite_budget_count: int | None
    satellite_budget_mode: str | None
    comparison_group_id: str | None
    prefill_pool_satellites: int | None
    decode_pool_satellites: int | None
    kv_policy_id: str | None
    kv_serve_actions_enabled: tuple[str, ...]
    kv_post_turn_actions_enabled: tuple[str, ...]
    replica_count_limit: int | None
    replica_target_deployment_id: str | None
    kv_page_size_tokens: int | None
    kv_transfer_chunk_bytes: float | None
    eviction_policy: str | None
    kv_hbm_model: str | None
    kv_transfer_model: str | None
    catalog_rebuilt: bool
    catalog_builder: str | None
    catalog_warnings: tuple[str, ...]
    scheduler: dict[str, Any]
    runtime_evaluated: bool
    objective_score: float | None
    selected: bool
    pareto_member: bool
    objective_score_scope: str | None
    valid_for_paper_main_results: bool
    cost_total_usd: float | None
    annualized_cost_usd_per_year: float | None
    served_ratio: float | None
    leo_served_ratio: float | None
    rejected_ratio: float | None
    timeout_ratio: float | None
    ttft_p95_s: float | None
    ttft_p99_s: float | None
    e2e_p95_s: float | None
    e2e_p99_s: float | None
    queue_delay_p95_s: float | None
    link_delay_p95_s: float | None
    useful_request_throughput_per_s: float | None
    useful_token_throughput_per_s: float | None
    cost_per_served_request_usd: float | None
    cost_per_served_mtoken_usd: float | None
    energy_violation_count: int | None
    fallback_request_fraction: float | None
    deferred_request_fraction: float | None
    shadow_affected_request_fraction: float | None
    shadow_served_ratio: float | None
    shadow_dvfs_request_fraction: float | None
    shadow_offload_request_fraction: float | None
    shadow_local_compute_request_fraction: float | None
    cross_satellite_forward_request_fraction: float | None
    trace_model: str | None
    scope: str | None
    runtime_result: dict[str, Any]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class DSERuntimeMatrixResult:
    evaluated: bool = False
    analysis_scope: str = "not_evaluated"
    design_count: int = 0
    policy_count: int = 0
    placement_scope_candidate_count: int = 0
    row_count: int = 0
    matrix_rows: tuple[DSERuntimeMatrixRow, ...] = ()
    selected_rows: tuple[dict[str, Any], ...] = ()
    selected_by_design: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_scope_by_design: dict[str, dict[str, Any]] = field(default_factory=dict)
    pareto_table: tuple[dict[str, Any], ...] = ()
    metrics: DSEMetrics = field(default_factory=DSEMetrics)
    static_screening: dict[str, Any] = field(default_factory=dict)
    non_separability: dict[str, Any] = field(default_factory=dict)
    allow_preliminary_runtime_metrics: bool = False
    allow_idealized_runtime_metrics: bool = False
    ignore_optional_cost: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["matrix_rows"] = [row.to_dict() for row in self.matrix_rows]
        payload["metrics"] = self.metrics.to_dict()
        return payload


def unevaluated_runtime_matrix(
    *,
    allow_preliminary_runtime_metrics: bool = False,
    allow_idealized_runtime_metrics: bool = False,
    ignore_optional_cost: bool = False,
) -> DSERuntimeMatrixResult:
    allow_preliminary = bool(allow_preliminary_runtime_metrics or allow_idealized_runtime_metrics)
    warnings = [
        "runtime scheduler matrix is disabled",
    ]
    if not allow_preliminary:
        warnings.append(
            "preliminary model runtime metrics are excluded from formal Pareto/non-separability analysis"
        )
    return DSERuntimeMatrixResult(
        allow_preliminary_runtime_metrics=allow_preliminary,
        allow_idealized_runtime_metrics=allow_idealized_runtime_metrics,
        ignore_optional_cost=ignore_optional_cost,
        warnings=tuple(warnings),
        provenance={"phase": "dse_runtime_matrix", "status": "disabled"},
    )


def evaluate_runtime_matrix(
    contexts: tuple[DSEDesignContext, ...] | list[DSEDesignContext],
    spec: Any,
) -> DSERuntimeMatrixResult:
    """Evaluate the design x runtime-policy matrix without invoking the optimizer."""

    context_tuple = tuple(contexts)
    static_infos = _static_design_infos(context_tuple, spec)
    screening = _select_static_screened_contexts(context_tuple, static_infos, spec)
    evaluated_contexts = screening["selected_contexts"]
    allow_preliminary = bool(
        getattr(spec, "allow_preliminary_runtime_metrics", False)
        or getattr(spec, "allow_surrogate_runtime_metrics", False)
        or getattr(spec, "allow_sanity_runtime_metrics", False)
        or getattr(spec, "allow_idealized_runtime_metrics", False)
    )
    ignore_optional_cost = bool(getattr(spec, "ignore_optional_cost", False))
    objective = _objective_from_spec(
        spec,
        allow_preliminary=allow_preliminary,
        ignore_optional_cost=ignore_optional_cost,
    )
    draft_rows: list[DSERuntimeMatrixRow] = []
    scope_candidates = _placement_scope_candidates(spec)
    candidate_cache = {
        context.design_id: _candidate_schedulers(context, spec)
        for context in context_tuple
    }
    policy_ids = {
        str(candidate.get("strategy_id") or candidate.get("policy"))
        for candidates in candidate_cache.values()
        for candidate in candidates
    }

    # A terrestrial service is one external reference system, not one row per
    # LEO design or PD placement scope.  Evaluate every unique configured
    # terrestrial policy once, before static LEO design screening.
    terrestrial_references: dict[
        str, tuple[DSEDesignContext, dict[str, Any]]
    ] = {}
    for context in context_tuple:
        for candidate in candidate_cache.get(context.design_id, ()):
            if not _is_terrestrial_policy(candidate.get("policy")):
                continue
            key_payload = {
                "policy": candidate.get("policy"),
                "ground_config": candidate.get("ground_config") or {},
                "path_policy": candidate.get("path_policy"),
                "queue_policy": candidate.get("queue_policy"),
                "max_hop_count": candidate.get("max_hop_count"),
                "workload_fingerprint": _terrestrial_workload_fingerprint(
                    context.runtime_scenario
                ),
            }
            key = json.dumps(to_jsonable(key_payload), sort_keys=True, separators=(",", ":"))
            terrestrial_references.setdefault(key, (context, dict(candidate)))

    for source_context, candidate in terrestrial_references.values():
        scheduler_payload = dict(candidate)
        base_strategy_id = str(scheduler_payload.pop("strategy_id"))
        strategy = RuntimeStrategySpec.from_dict(scheduler_payload)
        reference_context = _terrestrial_reference_context(
            source_context, policy=strategy.policy
        )
        runtime_result = _runtime_result_with_cost_efficiency(
            _compact_runtime_result(
                evaluate_runtime_performance(reference_context.runtime_scenario, strategy)
            ),
            context=reference_context,
            scheduler=scheduler_payload,
        )
        scoring = score_runtime_evaluator_result(runtime_result, objective)
        row = _matrix_row_from_runtime_result(
            context=reference_context,
            strategy_id=f"terrestrial_reference::{base_strategy_id}",
            placement_scope_strategy_id=None,
            scope_candidate={"current_scope": True, "terrestrial_reference": True},
            scope_candidate_count=0,
            static_info={},
            static_screen_reason="not_applicable_terrestrial_reference",
            catalog_preview=None,
            scheduler={**strategy.to_dict(), "kv_policy_id": None},
            runtime_result=runtime_result,
            scoring=scoring,
        )
        draft_rows.append(
            replace(
                row,
                static_screen_selected=False,
                static_candidate_selected=False,
                static_screen_reason="not_applicable_terrestrial_reference",
            )
        )

    for context in evaluated_contexts:
        candidates = tuple(
            candidate
            for candidate in candidate_cache.get(context.design_id, ())
            if not _is_terrestrial_policy(candidate.get("policy"))
        )
        for scope_candidate in scope_candidates:
            scoped_scenario = _runtime_scenario_for_scope_candidate(context.runtime_scenario, scope_candidate)
            scope_strategy_id = _optional_str(scope_candidate.get("strategy_id"))
            catalog_preview = build_runtime_deployment_catalog(scoped_scenario)
            for scheduler in candidates:
                scheduler_payload = dict(scheduler)
                base_strategy_id = str(scheduler_payload.pop("strategy_id"))
                strategy_id = _matrix_strategy_id(base_strategy_id, scope_strategy_id, scope_candidate)
                strategy = RuntimeStrategySpec.from_dict(scheduler_payload)
                full_runtime_result = evaluate_runtime_performance(scoped_scenario, strategy)
                runtime_result = _runtime_result_with_cost_efficiency(
                    _compact_runtime_result(full_runtime_result),
                    context=context,
                    scheduler=scheduler_payload,
                )
                scoring = score_runtime_evaluator_result(runtime_result, objective)
                row = _matrix_row_from_runtime_result(
                    context=context,
                    strategy_id=strategy_id,
                    placement_scope_strategy_id=scope_strategy_id,
                    scope_candidate=scope_candidate,
                    scope_candidate_count=len(scope_candidates),
                    static_info=static_infos.get(context.design_id, {}),
                    static_screen_reason=_optional_str(
                        _mapping_from_value(screening.get("selection_reasons")).get(context.design_id)
                    ),
                    catalog_preview=catalog_preview,
                    scheduler={**strategy.to_dict(), "kv_policy_id": scheduler_payload.get("kv_policy_id")},
                    runtime_result=runtime_result,
                    scoring=scoring,
                )
                draft_rows.append(row)
    context_by_design = {context.design_id: context for context in context_tuple}
    for skipped in tuple(screening.get("skipped_designs", ()) or ()):
        skipped_payload = _mapping_from_value(skipped)
        design_id = _optional_str(skipped_payload.get("design_id"))
        context = context_by_design.get(design_id or "")
        if context is None:
            continue
        draft_rows.append(
            _skipped_matrix_row(
                context=context,
                static_info=static_infos.get(context.design_id, {}),
                reason=_optional_str(skipped_payload.get("reason")) or "not_selected_by_static_screening",
            )
        )
    selected_keys = _selected_keys_by_design(draft_rows)
    pareto_keys = _pareto_keys(draft_rows)
    rows = tuple(
        replace(
            row,
            selected=(row.design_id, row.strategy_id) in selected_keys,
            pareto_member=(row.design_id, row.strategy_id) in pareto_keys,
        )
        for row in draft_rows
    )
    selected_rows = tuple(row.to_dict() for row in rows if row.selected)
    selected_by_design = {row["design_id"]: row for row in selected_rows}
    selected_scope_by_design = {
        row["design_id"]: {
            "placement_scope_strategy_id": row.get("placement_scope_strategy_id"),
            "pd_placement_scope": row.get("pd_placement_scope"),
            "pd_pairing_rule": row.get("pd_pairing_rule"),
        }
        for row in selected_rows
    }
    pareto_table = tuple(
        _pareto_payload(row)
        for row in rows
        if row.pareto_member
    )
    non_separability = _non_separability_summary(
        rows=rows,
        static_infos=static_infos,
        screening=screening,
    )
    warnings = _matrix_warnings(
        rows,
        allow_preliminary=allow_preliminary,
        screening=screening,
        non_separability=non_separability,
    )
    matrix_metrics = DSEMetrics(
        evaluated=bool(rows),
        pareto_frontier_evaluated=bool(pareto_table),
        non_separability_regret=_optional_float(non_separability.get("regret_served_ratio")),
        dominated_design_count=_optional_int(non_separability.get("dominated_design_count")),
        rank_disagreement_count=_optional_int(non_separability.get("rank_disagreement_count")),
        warnings=(
            "runtime_matrix.pareto_table is preliminary-model analysis and not the formal paper-main DSE Pareto frontier",
            *tuple(non_separability.get("warnings", ()) or ()),
        ),
    )
    return DSERuntimeMatrixResult(
        evaluated=True,
        analysis_scope="preliminary_model" if allow_preliminary else "strict_unscored",
        design_count=len(context_tuple),
        policy_count=len(policy_ids),
        placement_scope_candidate_count=len(scope_candidates),
        row_count=len(rows),
        matrix_rows=rows,
        selected_rows=selected_rows,
        selected_by_design=selected_by_design,
        selected_scope_by_design=selected_scope_by_design,
        pareto_table=pareto_table,
        metrics=matrix_metrics,
        static_screening=_static_screening_payload(screening, static_infos),
        non_separability=non_separability,
        allow_preliminary_runtime_metrics=allow_preliminary,
        allow_idealized_runtime_metrics=bool(getattr(spec, "allow_idealized_runtime_metrics", False)),
        ignore_optional_cost=ignore_optional_cost,
        provenance={
            "phase": "dse_runtime_matrix",
            "runtime_evaluator": "orbitserve.runtime.evaluate_runtime_performance",
            "optimizer_called": False,
            "scorer": "orbitserve.dse.runtime_optimization.score_runtime_evaluator_result",
            "static_screening_enabled": bool(screening["config"].get("enabled", False)),
            "valid_for_paper_main_results": False,
            "pareto_table_scope": "preliminary_runtime_matrix",
        },
        warnings=warnings,
    )


def _static_design_infos(
    contexts: tuple[DSEDesignContext, ...],
    spec: Any,
) -> dict[str, dict[str, Any]]:
    weights = _static_screening_config(spec).get("weights") or _mapping_from_value(
        getattr(spec, "objective", {})
    ).get("weights") or {"served": 1.0, "tail": 1.0, "cost": 0.0}
    infos: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        annualized_cost = _optional_float(
            _design_row_metric(context.design_row, "cost", "annualized_cost_usd_per_year")
        )
        served_ratio = _optional_float(_design_row_metric(context.design_row, "slo", "served_ratio"))
        e2e_p95_s = _optional_float(_design_row_metric(context.design_row, "performance", "e2e_p95_s"))
        ttft_p95_s = _optional_float(_design_row_metric(context.design_row, "performance", "ttft_p95_s"))
        static_score = _static_objective_score(
            served_ratio=served_ratio,
            e2e_p95_s=e2e_p95_s,
            annualized_cost_usd_per_year=annualized_cost,
            weights=_mapping_from_value(weights),
        )
        infos.append(
            {
                "design_id": context.design_id,
                "input_index": index,
                "static_score": static_score,
                "served_ratio": served_ratio,
                "e2e_p95_s": e2e_p95_s,
                "ttft_p95_s": ttft_p95_s,
                "annualized_cost_usd_per_year": annualized_cost,
                "rank": None,
                "frontier_member": False,
                "rankable": static_score is not None,
            }
        )

    ranked = sorted(
        (info for info in infos if info["static_score"] is not None),
        key=lambda item: (
            float(item["static_score"]),
            float(item["annualized_cost_usd_per_year"])
            if item["annualized_cost_usd_per_year"] is not None
            else float("inf"),
            str(item["design_id"]),
        ),
    )
    for rank, info in enumerate(ranked, start=1):
        info["rank"] = rank
    frontier_ids = _static_frontier_ids(tuple(infos))
    for info in infos:
        info["frontier_member"] = info["design_id"] in frontier_ids
    return {str(info["design_id"]): info for info in infos}


def _design_row_metric(row: Any, section: str, field_name: str) -> Any:
    """Read a scalar metric without serializing bulky design-row detail payloads."""

    if isinstance(row, Mapping):
        section_value = row.get(section)
    else:
        section_value = getattr(row, section, None)
    if isinstance(section_value, Mapping):
        return section_value.get(field_name)
    return getattr(section_value, field_name, None)


def _static_screening_config(spec: Any) -> dict[str, Any]:
    closed_loop = _mapping_from_value(getattr(spec, "closed_loop", None))
    closed_loop_screening = _mapping_from_value(closed_loop.get("static_screening"))
    raw_static = _mapping_from_value(
        getattr(spec, "static_screening", None)
        or getattr(spec, "static_to_runtime", None)
        or getattr(spec, "static_screen", None)
        or {}
    )
    raw = {
        **closed_loop,
        **closed_loop_screening,
        **raw_static,
    }
    top_k = _optional_int(
        raw.get("top_k_designs")
        if raw.get("top_k_designs") is not None
        else raw.get("top_k")
    )
    enabled_value = _first_present(
        raw_static.get("enabled"),
        raw_static.get("static_screening_enabled"),
        closed_loop_screening.get("enabled"),
        closed_loop_screening.get("static_screening_enabled"),
        closed_loop.get("static_screening_enabled"),
    )
    enabled = bool(enabled_value) if enabled_value is not None else top_k is not None
    return {
        **raw,
        "enabled": enabled,
        "top_k_designs": top_k,
        "include_static_best": bool(raw.get("include_static_best", True)),
        "include_static_frontier": bool(raw.get("include_static_frontier", False)),
        "prefer_static_candidates": bool(raw.get("prefer_static_candidates", True)),
        "fail_open_on_unrankable": bool(raw.get("fail_open_on_unrankable", True)),
        "always_include_design_ids": tuple(str(item) for item in raw.get("always_include_design_ids", ()) or ()),
    }


def _select_static_screened_contexts(
    contexts: tuple[DSEDesignContext, ...],
    static_infos: Mapping[str, Mapping[str, Any]],
    spec: Any,
) -> dict[str, Any]:
    config = _static_screening_config(spec)
    selected_ids: set[str] = set()
    reasons: dict[str, str] = {}
    if not config["enabled"]:
        for context in contexts:
            selected_ids.add(context.design_id)
            reasons[context.design_id] = "static_screening_disabled"
        return {
            "config": config,
            "selected_contexts": contexts,
            "skipped_contexts": (),
            "selected_design_ids": tuple(context.design_id for context in contexts),
            "skipped_designs": (),
            "selection_reasons": reasons,
            "by_design": {
                context.design_id: {"selected": True, "reason": reasons[context.design_id]}
                for context in contexts
            },
            "warnings": (),
        }

    ranked_infos = sorted(
        (
            info
            for info in static_infos.values()
            if info.get("rank") is not None
        ),
        key=lambda item: int(item["rank"]),
    )
    if not ranked_infos and config["fail_open_on_unrankable"]:
        for context in contexts:
            selected_ids.add(context.design_id)
            reasons[context.design_id] = "static_screening_fail_open_unrankable"
        return {
            "config": config,
            "selected_contexts": contexts,
            "skipped_contexts": (),
            "selected_design_ids": tuple(context.design_id for context in contexts),
            "skipped_designs": (),
            "selection_reasons": reasons,
            "by_design": {
                context.design_id: {"selected": True, "reason": reasons[context.design_id]}
                for context in contexts
            },
            "warnings": ("static screening could not rank designs; evaluating all designs",),
        }

    top_k = config["top_k_designs"]
    if top_k is not None and top_k > 0:
        for info in ranked_infos[:top_k]:
            selected_ids.add(str(info["design_id"]))
            reasons[str(info["design_id"])] = f"static_top_{top_k}"
    if config["include_static_best"] and ranked_infos:
        best = str(ranked_infos[0]["design_id"])
        selected_ids.add(best)
        reasons.setdefault(best, "static_best")
    if config["include_static_frontier"]:
        for info in static_infos.values():
            if bool(info.get("frontier_member")):
                design_id = str(info["design_id"])
                selected_ids.add(design_id)
                reasons.setdefault(design_id, "static_frontier")
    for design_id in config["always_include_design_ids"]:
        selected_ids.add(design_id)
        reasons.setdefault(design_id, "always_include_design_id")
    if not selected_ids and contexts:
        first = contexts[0].design_id
        selected_ids.add(first)
        reasons[first] = "static_screening_empty_fallback_first_design"

    selected_contexts = tuple(context for context in contexts if context.design_id in selected_ids)
    skipped_contexts = tuple(context for context in contexts if context.design_id not in selected_ids)
    skipped = tuple(
        {
            "design_id": context.design_id,
            "static_rank": _optional_int(static_infos.get(context.design_id, {}).get("rank")),
            "static_score": _optional_float(static_infos.get(context.design_id, {}).get("static_score")),
            "reason": "not_selected_by_static_screening",
        }
        for context in skipped_contexts
    )
    return {
        "config": config,
        "selected_contexts": selected_contexts,
        "skipped_contexts": skipped_contexts,
        "selected_design_ids": tuple(context.design_id for context in selected_contexts),
        "skipped_designs": skipped,
        "selection_reasons": reasons,
        "by_design": {
            context.design_id: {
                "selected": context.design_id in selected_ids,
                "reason": reasons.get(context.design_id, "not_selected_by_static_screening"),
            }
            for context in contexts
        },
        "warnings": (),
    }


def _static_screening_payload(
    screening: Mapping[str, Any],
    static_infos: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "enabled": bool(_mapping_from_value(screening.get("config")).get("enabled", False)),
        "config": to_jsonable(screening.get("config")),
        "selected_design_ids": list(screening.get("selected_design_ids", ()) or ()),
        "selected_count": len(tuple(screening.get("selected_design_ids", ()) or ())),
        "skipped_count": len(tuple(screening.get("skipped_designs", ()) or ())),
        "skipped_designs": to_jsonable(screening.get("skipped_designs", ())),
        "selection_reasons": to_jsonable(screening.get("selection_reasons", {})),
        "static_designs": tuple(to_jsonable(info) for info in static_infos.values()),
        "warnings": list(screening.get("warnings", ()) or ()),
    }


def _static_objective_score(
    *,
    served_ratio: float | None,
    e2e_p95_s: float | None,
    annualized_cost_usd_per_year: float | None,
    weights: Mapping[str, Any],
) -> float | None:
    if served_ratio is None or e2e_p95_s is None:
        return None
    score = -float(weights.get("served", 1.0)) * served_ratio + float(weights.get("tail", 1.0)) * e2e_p95_s
    cost_weight = float(weights.get("cost", 0.0) or 0.0)
    if cost_weight and annualized_cost_usd_per_year is not None:
        score += cost_weight * annualized_cost_usd_per_year
    return score


def _static_frontier_ids(infos: tuple[Mapping[str, Any], ...]) -> set[str]:
    eligible = tuple(info for info in infos if info.get("served_ratio") is not None and info.get("e2e_p95_s") is not None)
    frontier: set[str] = set()
    for candidate in eligible:
        if not any(_static_dominates(other, candidate) for other in eligible if other is not candidate):
            frontier.add(str(candidate["design_id"]))
    return frontier


def _static_dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    comparisons: list[tuple[float, float, str]] = []
    for field, direction in (
        ("annualized_cost_usd_per_year", "lower"),
        ("served_ratio", "higher"),
        ("e2e_p95_s", "lower"),
    ):
        left_value = _optional_float(left.get(field))
        right_value = _optional_float(right.get(field))
        if left_value is not None and right_value is not None:
            comparisons.append((left_value, right_value, direction))
    if not comparisons:
        return False
    no_worse = all(a <= b if direction == "lower" else a >= b for a, b, direction in comparisons)
    strictly_better = any(a < b if direction == "lower" else a > b for a, b, direction in comparisons)
    return no_worse and strictly_better


def _non_separability_summary(
    *,
    rows: tuple[DSERuntimeMatrixRow, ...],
    static_infos: Mapping[str, Mapping[str, Any]],
    screening: Mapping[str, Any],
) -> dict[str, Any]:
    selected_rows = tuple(row for row in rows if row.selected and row.objective_score is not None)
    if not selected_rows:
        return {
            "evaluated": False,
            "warnings": ("non-separability metrics require selected scored runtime rows",),
        }
    static_ranked_ids = [
        str(info["design_id"])
        for info in sorted(
            (info for info in static_infos.values() if info.get("rank") is not None),
            key=lambda item: int(item["rank"]),
        )
    ]
    runtime_ranked_rows = sorted(
        selected_rows,
        key=lambda row: (
            float(row.objective_score) if row.objective_score is not None else float("inf"),
            str(row.design_id),
        ),
    )
    runtime_rank_by_design = {row.design_id: index for index, row in enumerate(runtime_ranked_rows, start=1)}
    comparable_ids = [design_id for design_id in static_ranked_ids if design_id in runtime_rank_by_design]
    incomplete_ids = [design_id for design_id in static_ranked_ids if design_id not in runtime_rank_by_design]
    rank_comparison_complete = not incomplete_ids
    pair_disagreement: int | None = 0 if rank_comparison_complete else None
    if rank_comparison_complete:
        for left_index, left_id in enumerate(comparable_ids):
            for right_id in comparable_ids[left_index + 1 :]:
                static_order = static_ranked_ids.index(left_id) < static_ranked_ids.index(right_id)
                runtime_order = runtime_rank_by_design[left_id] < runtime_rank_by_design[right_id]
                if static_order != runtime_order:
                    pair_disagreement = int(pair_disagreement or 0) + 1

    static_best_id = static_ranked_ids[0] if static_ranked_ids else None
    comparable_static_best_id = comparable_ids[0] if comparable_ids else None
    runtime_best = runtime_ranked_rows[0]
    static_best_runtime = next((row for row in selected_rows if row.design_id == static_best_id), None)
    static_best_runtime_attempt = next(
        (row for row in rows if row.design_id == static_best_id and row.policy != "static_screen_skipped"),
        None,
    )
    regret_served_ratio = None
    regret_e2e_p95_s = None
    if static_best_runtime is not None:
        if runtime_best.served_ratio is not None and static_best_runtime.served_ratio is not None:
            regret_served_ratio = max(0.0, float(runtime_best.served_ratio) - float(static_best_runtime.served_ratio))
        if runtime_best.e2e_p95_s is not None and static_best_runtime.e2e_p95_s is not None:
            regret_e2e_p95_s = max(0.0, float(static_best_runtime.e2e_p95_s) - float(runtime_best.e2e_p95_s))
    dominated_design_count = sum(
        1
        for row in selected_rows
        if any(_dominates(other, row) for other in selected_rows if other is not row)
    )
    best_differs = static_best_id is not None and static_best_id != runtime_best.design_id

    warnings: list[str] = []
    skipped_count = len(tuple(screening.get("skipped_designs", ()) or ()))
    if skipped_count:
        warnings.append("non-separability metrics compare only designs admitted by static screening")
    if incomplete_ids:
        warnings.append("rank disagreement is null because some static-ranked designs have no scored runtime row")
    if static_best_id is None:
        warnings.append("static best design is unavailable because static rows were unrankable")
    elif static_best_runtime is None:
        warnings.append("static best design has no scored runtime row; regret metrics are null")
    return {
        "evaluated": True,
        "scope": "static_screened_runtime_matrix",
        "analysis": (
            "static best and runtime best differ under preliminary runtime scoring"
            if best_differs
            else "static best and runtime best match under preliminary runtime scoring"
        ),
        "static_best_design_id": static_best_id,
        "comparable_static_best_design_id": comparable_static_best_id,
        "runtime_best_design_id": runtime_best.design_id,
        "static_runtime_best_differs": best_differs,
        "runtime_best_strategy_id": runtime_best.strategy_id,
        "runtime_best_policy": runtime_best.policy,
        "static_best_runtime_strategy_id": static_best_runtime.strategy_id if static_best_runtime else None,
        "rank_disagreement_count": pair_disagreement,
        "rank_comparison_complete": rank_comparison_complete,
        "rank_comparison_missing_design_ids": tuple(incomplete_ids),
        "rank_disagreement_scope": "pairwise_inversions_between_static_rank_and_runtime_selected_rank",
        "regret_served_ratio": regret_served_ratio,
        "non_separability_regret": regret_served_ratio,
        "regret_e2e_p95_s": regret_e2e_p95_s,
        "dominated_design_count": dominated_design_count,
        "static_ranked_design_ids": static_ranked_ids,
        "comparable_static_ranked_design_ids": comparable_ids,
        "runtime_ranked_design_ids": [row.design_id for row in runtime_ranked_rows],
        "screened_design_count": len(tuple(screening.get("selected_design_ids", ()) or ())),
        "runtime_scored_design_count": len(selected_rows),
        "skipped_design_count": skipped_count,
        "valid_for_paper_main_results": False,
        "static_best": _static_info_summary(static_infos.get(static_best_id, {}) if static_best_id else {}),
        "runtime_best": _runtime_row_summary(runtime_best),
        "static_best_runtime": _runtime_row_summary(static_best_runtime) if static_best_runtime else {},
        "static_best_runtime_attempt": _runtime_row_summary(static_best_runtime_attempt)
        if static_best_runtime_attempt
        else {},
        "provenance": {
            "source": "orbitserve.dse_runtime_matrix._non_separability_summary",
            "static_rank_source": "DSEDesignRow.slo/performance static_slo_proxy",
            "runtime_rank_source": "selected runtime_matrix rows scored by runtime_optimization objective",
            "screening_config": to_jsonable(screening.get("config")),
            "valid_for_paper_main_results": False,
        },
        "warnings": tuple(warnings),
    }


def _static_info_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "design_id": info.get("design_id"),
        "static_rank": _optional_int(info.get("rank")),
        "static_score": _optional_float(info.get("static_score")),
        "static_frontier_member": bool(info.get("frontier_member", False)),
        "static_served_ratio": _optional_float(info.get("served_ratio")),
        "static_e2e_p95_s": _optional_float(info.get("e2e_p95_s")),
        "static_annualized_cost_usd_per_year": _optional_float(info.get("annualized_cost_usd_per_year")),
    }


def _runtime_row_summary(row: DSERuntimeMatrixRow) -> dict[str, Any]:
    return {
        "design_id": row.design_id,
        "strategy_id": row.strategy_id,
        "policy": row.policy,
        "objective_score": row.objective_score,
        "served_ratio": row.served_ratio,
        "leo_served_ratio": row.leo_served_ratio,
        "e2e_p95_s": row.e2e_p95_s,
        "e2e_p99_s": row.e2e_p99_s,
        "annualized_cost_usd_per_year": row.annualized_cost_usd_per_year,
        "cost_per_served_mtoken_usd": row.cost_per_served_mtoken_usd,
        "valid_for_paper_main_results": row.valid_for_paper_main_results,
    }


def _objective_from_spec(
    spec: Any,
    *,
    allow_preliminary: bool,
    ignore_optional_cost: bool,
) -> RuntimeOptimizationObjective:
    payload = dict(getattr(spec, "objective", {}) or {})
    payload["allow_preliminary_runtime_metrics"] = bool(
        allow_preliminary
        or payload.get("allow_preliminary_runtime_metrics", False)
        or payload.get("allow_surrogate_runtime_metrics", False)
        or payload.get("allow_sanity_runtime_metrics", False)
    )
    payload["allow_idealized_runtime_metrics"] = bool(
        getattr(spec, "allow_idealized_runtime_metrics", False)
        or payload.get("allow_idealized_runtime_metrics", False)
    )
    payload["ignore_optional_cost"] = bool(
        ignore_optional_cost or payload.get("ignore_optional_cost", False)
    )
    return RuntimeOptimizationObjective.from_dict(payload)


def _candidate_schedulers(context: DSEDesignContext, spec: Any) -> tuple[dict[str, Any], ...]:
    base_candidates: list[dict[str, Any]] = []
    if bool(getattr(spec, "include_current_policy", True)):
        current = RuntimeStrategySpec.from_dict(context.current_scheduler).to_dict()
        _append_candidate(base_candidates, current, strategy_id=f"current_{current['policy']}")
    for policy in _string_tuple(getattr(spec, "policies", ())):
        policy_payload: dict[str, Any] = {"policy": policy}
        if _is_terrestrial_policy(policy):
            ground_config = _mapping_from_value(getattr(spec, "ground_config", None))
            if not ground_config and _is_terrestrial_policy(
                _mapping_from_value(context.current_scheduler).get("policy")
            ):
                ground_config = _mapping_from_value(
                    _mapping_from_value(context.current_scheduler).get("ground_config")
                )
            if ground_config:
                policy_payload["ground_config"] = ground_config
        scheduler = RuntimeStrategySpec.from_dict(policy_payload).to_dict()
        _append_candidate(base_candidates, scheduler, strategy_id=str(scheduler["policy"]))
    for index, raw_scheduler in enumerate(getattr(spec, "candidate_schedulers", ()) or ()):
        scheduler = RuntimeStrategySpec.from_dict(_mapping_from_value(raw_scheduler)).to_dict()
        strategy_id = str(_mapping_from_value(raw_scheduler).get("strategy_id") or scheduler["policy"])
        _append_candidate(base_candidates, scheduler, strategy_id=strategy_id, fallback_suffix=index)
    kv_candidates = _kv_policy_candidates(spec)
    expanded: list[dict[str, Any]] = []
    for scheduler in base_candidates:
        if _is_terrestrial_policy(scheduler.get("policy")):
            # Orbital KV policy search dimensions do not apply to the
            # independent terrestrial reference.
            expanded.append(to_jsonable(scheduler))
            continue
        for kv_candidate in kv_candidates:
            payload = dict(scheduler)
            base_strategy_id = str(payload.get("strategy_id"))
            kv_policy = _mapping_from_value(kv_candidate.get("kv_policy"))
            if kv_policy:
                payload["kv_policy"] = kv_policy
            kv_policy_id = _optional_str(kv_candidate.get("kv_policy_id"))
            if kv_policy_id is not None:
                payload["kv_policy_id"] = kv_policy_id
                payload["strategy_id"] = f"{base_strategy_id}::kv_{kv_policy_id}"
            expanded.append(to_jsonable(payload))
    return tuple(expanded)


def _kv_policy_candidates(spec: Any) -> tuple[dict[str, Any], ...]:
    raw_candidates = tuple(
        getattr(spec, "kv_policy_candidates", ())
        or _mapping_from_value(getattr(spec, "kv_policy_search_space", None)).get("kv_policy_candidates", ())
        or ()
    )
    if not raw_candidates:
        return ({"kv_policy_id": None, "kv_policy": None},)
    candidates: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(raw_candidates):
        payload = _mapping_from_value(raw_candidate)
        kv_policy = _mapping_from_value(payload.get("kv_policy")) or {
            key: value
            for key, value in payload.items()
            if key
            in {
                "kv_serve_actions_enabled",
                "kv_post_turn_actions_enabled",
                "replica_count_limit",
                "replica_target_deployment_id",
                "session_ttl_s",
                "kv_page_size_tokens",
                "kv_page_metadata_bytes",
                "kv_fragmentation_reserve_fraction",
                "kv_transfer_chunk_bytes",
                "max_kv_transfer_hop_count",
                "eviction_policy",
                "hbm_model",
            }
        }
        kv_policy_id = (
            _optional_str(payload.get("kv_policy_id"))
            or _optional_str(payload.get("strategy_id"))
            or f"kv_policy_{index}"
        )
        candidates.append({"kv_policy_id": kv_policy_id, "kv_policy": kv_policy})
    return tuple(candidates)


def _append_candidate(
    candidates: list[dict[str, Any]],
    scheduler: dict[str, Any],
    *,
    strategy_id: str,
    fallback_suffix: int | None = None,
) -> None:
    candidate = dict(scheduler)
    candidate["strategy_id"] = strategy_id
    seen = {item["strategy_id"] for item in candidates}
    if candidate["strategy_id"] in seen:
        if fallback_suffix is None:
            return
        candidate["strategy_id"] = f"{candidate['strategy_id']}_{fallback_suffix}"
    candidates.append(to_jsonable(candidate))


def _placement_scope_candidates(spec: Any) -> tuple[dict[str, Any], ...]:
    raw_candidates = tuple(
        getattr(spec, "placement_scope_candidates", ())
        or getattr(spec, "scope_candidates", ())
        or _mapping_from_value(getattr(spec, "placement_scope_search_space", None)).get("scope_candidates", ())
        or ()
    )
    if not raw_candidates:
        return ({"strategy_id": None, "pd_placement": None, "current_scope": True},)
    candidates: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(raw_candidates):
        payload = _mapping_from_value(raw_candidate)
        pd_placement = _mapping_from_value(payload.get("pd_placement"))
        placement_scope = _optional_str(payload.get("placement_scope") or payload.get("pd_placement_scope"))
        if placement_scope is not None:
            pd_placement.setdefault("placement_scope", placement_scope)
        for key in (
            "explicit_pairs",
            "pairs",
            "hybrid_template",
            "template",
            "hybrid_template_name",
            "max_pairs_per_prefill",
            "max_pairs_per_prefill_group",
            "max_pair_count",
            "max_cross_plane_pairs_per_prefill",
            "max_cross_plane_pairs_per_prefill_group",
            "fail_closed_on_unsupported_scope",
        ):
            if key in payload and key not in pd_placement:
                pd_placement[key] = payload[key]
        if not pd_placement:
            continue
        strategy_id = (
            _optional_str(payload.get("strategy_id"))
            or _optional_str(payload.get("placement_scope_strategy_id"))
            or _optional_str(pd_placement.get("placement_scope"))
            or f"pd_scope_{index}"
        )
        candidates.append(
            to_jsonable(
                {
                    **payload,
                    "strategy_id": strategy_id,
                    "placement_scope_strategy_id": strategy_id,
                    "pd_placement": pd_placement,
                    "current_scope": False,
                }
            )
        )
    return tuple(candidates) or ({"strategy_id": None, "pd_placement": None, "current_scope": True},)


def _runtime_scenario_for_scope_candidate(runtime_scenario: Any, scope_candidate: Mapping[str, Any]) -> Any:
    pd_placement = _mapping_from_value(scope_candidate.get("pd_placement"))
    if not pd_placement:
        return runtime_scenario
    deployments = tuple(
        _deployment_with_pd_placement(deployment, pd_placement=pd_placement)
        for deployment in tuple(getattr(runtime_scenario, "serving_deployments", ()) or ())
    )
    provenance = {
        **_mapping_from_value(getattr(runtime_scenario, "provenance", {})),
        "runtime_matrix_scope_candidate": {
            "strategy_id": scope_candidate.get("strategy_id"),
            "pd_placement": pd_placement,
            "catalog_rebuild_required": True,
        },
    }
    try:
        return replace(
            runtime_scenario,
            scenario_id=f"{getattr(runtime_scenario, 'scenario_id', 'runtime_scenario')}:{scope_candidate.get('strategy_id')}",
            serving_deployments=deployments,
            provenance=provenance,
        )
    except TypeError:
        pass
    if hasattr(runtime_scenario, "__dict__"):
        attrs = dict(vars(runtime_scenario))
        attrs["scenario_id"] = f"{getattr(runtime_scenario, 'scenario_id', 'runtime_scenario')}:{scope_candidate.get('strategy_id')}"
        attrs["serving_deployments"] = deployments
        attrs["provenance"] = provenance
        try:
            return type(runtime_scenario)(**attrs)
        except TypeError:
            pass
    return runtime_scenario


def _terrestrial_reference_context(
    context: DSEDesignContext, *, policy: str
) -> DSEDesignContext:
    scenario = context.runtime_scenario
    provenance = {
        **_mapping_from_value(getattr(scenario, "provenance", {})),
        "comparison_domain": "terrestrial_reference",
        "source_leo_design_id_ignored": context.design_id,
        "terrestrial_policy": policy,
    }
    replacement = {
        "scenario_id": f"terrestrial_reference:{policy}",
        "design_id": None,
        "constellation": {},
        "serving_deployments": (),
        "cost_power": {},
        "simulation_trace_ref": {},
        "provenance": provenance,
    }
    try:
        reference_scenario = replace(scenario, **replacement)
    except TypeError:
        attrs = dict(vars(scenario)) if hasattr(scenario, "__dict__") else {}
        attrs.update(replacement)
        try:
            reference_scenario = type(scenario)(**attrs)
        except TypeError:
            reference_scenario = scenario
            for key, value in replacement.items():
                try:
                    setattr(reference_scenario, key, value)
                except (AttributeError, TypeError):
                    pass
    return DSEDesignContext(
        design_id="terrestrial_reference",
        design_row={},
        runtime_scenario=reference_scenario,
        current_scheduler={"policy": policy},
        cost_result={},
    )


def _terrestrial_workload_fingerprint(runtime_scenario: Any) -> str:
    payload = {
        "requests": tuple(getattr(runtime_scenario, "requests", ()) or ()),
        "regions": tuple(getattr(runtime_scenario, "regions", ()) or ()),
        "gateways": tuple(getattr(runtime_scenario, "gateways", ()) or ()),
    }
    encoded = json.dumps(
        to_jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deployment_with_pd_placement(deployment: Any, *, pd_placement: dict[str, Any]) -> Any:
    serving_payload = _mapping_from_value(getattr(deployment, "serving_spec", None))
    throughput_payload = _mapping_from_value(getattr(deployment, "throughput_estimate", None))
    if not serving_payload and isinstance(deployment, Mapping):
        serving_payload = _mapping_from_value(deployment.get("serving_spec"))
    if not throughput_payload and isinstance(deployment, Mapping):
        throughput_payload = _mapping_from_value(deployment.get("throughput_estimate"))
    serving_payload = {**serving_payload, "pd_placement": pd_placement}
    throughput_payload = {
        **throughput_payload,
        "pd_placement": pd_placement,
        "pd_placement_scope": pd_placement.get("placement_scope"),
    }
    if hasattr(deployment, "to_dict") and hasattr(deployment, "serving_spec"):
        try:
            return replace(
                deployment,
                serving_spec=serving_payload,
                throughput_estimate=throughput_payload,
            )
        except TypeError:
            pass
    if hasattr(deployment, "__dict__"):
        attrs = dict(vars(deployment))
        attrs["serving_spec"] = serving_payload
        attrs["throughput_estimate"] = throughput_payload
        try:
            return type(deployment)(**attrs)
        except TypeError:
            pass
    payload = _mapping_from_value(deployment)
    return {
        **payload,
        "serving_spec": serving_payload,
        "throughput_estimate": throughput_payload,
        "pd_placement": pd_placement,
        "pd_placement_scope": pd_placement.get("placement_scope"),
    }


def _matrix_strategy_id(base_strategy_id: str, scope_strategy_id: str | None, scope_candidate: Mapping[str, Any]) -> str:
    if bool(scope_candidate.get("current_scope", False)) or scope_strategy_id is None:
        return base_strategy_id
    return f"{scope_strategy_id}::{base_strategy_id}"


def _runtime_result_with_cost_efficiency(
    runtime_result: dict[str, Any],
    *,
    context: DSEDesignContext,
    scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach annualized cost-per-served metrics when DSE cost context exists."""

    runtime_metrics = dict(_mapping_from_value(runtime_result.get("runtime_metrics")))
    cost_payload = _mapping_from_value(context.cost_result)
    policy = str(scheduler.get("policy") or "")
    terrestrial_baseline = _is_terrestrial_policy(policy)
    terrestrial_cost = _mapping_from_value(
        _mapping_from_value(runtime_result.get("details")).get("terrestrial_costing")
    )
    terrestrial_cost_evaluated = bool(terrestrial_cost.get("evaluated", False))
    annualized_cost = (
        _optional_float(terrestrial_cost.get("annualized_cost_usd_per_year"))
        if terrestrial_baseline and terrestrial_cost_evaluated
        else None
        if terrestrial_baseline
        else _optional_float(cost_payload.get("annualized_cost_usd_per_year"))
    )
    request_rate = _optional_float(runtime_metrics.get("useful_request_throughput_per_s"))
    token_rate = _optional_float(runtime_metrics.get("useful_token_throughput_per_s"))
    cost_per_request = (
        _optional_float(runtime_metrics.get("cost_per_served_request_usd"))
        if terrestrial_baseline and terrestrial_cost_evaluated
        else None
        if terrestrial_baseline
        else _cost_per_served_request(
            annualized_cost_usd_per_year=annualized_cost,
            useful_request_throughput_per_s=request_rate,
        )
    )
    cost_per_mtoken = (
        _optional_float(runtime_metrics.get("cost_per_served_mtoken_usd"))
        if terrestrial_baseline and terrestrial_cost_evaluated
        else None
        if terrestrial_baseline
        else _cost_per_served_mtoken(
            annualized_cost_usd_per_year=annualized_cost,
            useful_token_throughput_per_s=token_rate,
        )
    )
    if terrestrial_baseline:
        # The terrestrial cost result is generated solely from its explicit
        # ground_config contract.  Never substitute the surrounding LEO
        # design's capital cost when that contract is absent.
        runtime_metrics["cost_per_served_request_usd"] = cost_per_request
        runtime_metrics["cost_per_served_mtoken_usd"] = cost_per_mtoken
    else:
        if cost_per_request is not None:
            runtime_metrics["cost_per_served_request_usd"] = cost_per_request
        else:
            runtime_metrics.setdefault("cost_per_served_request_usd", None)
        if cost_per_mtoken is not None:
            runtime_metrics["cost_per_served_mtoken_usd"] = cost_per_mtoken
        else:
            runtime_metrics.setdefault("cost_per_served_mtoken_usd", None)

    slo = {
        **_mapping_from_value(runtime_result.get("slo")),
        "cost_per_served_request_usd": runtime_metrics.get("cost_per_served_request_usd"),
        "cost_per_served_mtoken_usd": runtime_metrics.get("cost_per_served_mtoken_usd"),
    }
    details = {
        **_mapping_from_value(runtime_result.get("details")),
        "cost_efficiency": {
            "annualized_cost_usd_per_year": annualized_cost,
            "useful_request_throughput_per_s": request_rate,
            "useful_token_throughput_per_s": token_rate,
            "cost_per_served_request_usd": runtime_metrics.get("cost_per_served_request_usd"),
            "cost_per_served_mtoken_usd": runtime_metrics.get("cost_per_served_mtoken_usd"),
            "basis": (
                "explicit_terrestrial_ground_costing_contract"
                if terrestrial_baseline and terrestrial_cost_evaluated and annualized_cost is not None
                else "terrestrial_cost_model_not_configured"
                if terrestrial_baseline
                else "annualized_cost divided by surrogate served throughput"
            ),
            "baseline_policy": policy if terrestrial_baseline else None,
            "valid_for_paper_main_results": False,
        },
    }
    if terrestrial_baseline and (not terrestrial_cost_evaluated or annualized_cost is None):
        warnings = tuple(runtime_result.get("warnings", ()) or ())
        runtime_result = {
            **runtime_result,
            "warnings": tuple(
                dict.fromkeys(
                    (
                        *warnings,
                        "ground reference has no configured terrestrial TCO; LEO capital cost is not copied into the baseline",
                    )
                )
            ),
        }
    return {
        **runtime_result,
        "runtime_metrics": runtime_metrics,
        "slo": slo,
        "details": details,
    }


def _cost_per_served_request(
    *,
    annualized_cost_usd_per_year: float | None,
    useful_request_throughput_per_s: float | None,
) -> float | None:
    annual_requests = _annualized_count(useful_request_throughput_per_s)
    if annualized_cost_usd_per_year is None or annual_requests is None:
        return None
    return annualized_cost_usd_per_year / annual_requests


def _cost_per_served_mtoken(
    *,
    annualized_cost_usd_per_year: float | None,
    useful_token_throughput_per_s: float | None,
) -> float | None:
    annual_tokens = _annualized_count(useful_token_throughput_per_s)
    if annualized_cost_usd_per_year is None or annual_tokens is None:
        return None
    annual_mtokens = annual_tokens / 1_000_000.0
    if annual_mtokens <= 0.0:
        return None
    return annualized_cost_usd_per_year / annual_mtokens


def _annualized_count(rate_per_s: float | None) -> float | None:
    if rate_per_s is None or rate_per_s <= 0.0:
        return None
    return rate_per_s * 365.0 * 24.0 * 3600.0


def _matrix_row_from_runtime_result(
    *,
    context: DSEDesignContext,
    strategy_id: str,
    placement_scope_strategy_id: str | None,
    scope_candidate: Mapping[str, Any],
    scope_candidate_count: int,
    static_info: Mapping[str, Any],
    static_screen_reason: str | None,
    catalog_preview: Any,
    scheduler: dict[str, Any],
    runtime_result: dict[str, Any],
    scoring: dict[str, Any],
) -> DSERuntimeMatrixRow:
    runtime_metrics = _mapping_from_value(runtime_result.get("runtime_metrics"))
    performance = _mapping_from_value(runtime_result.get("performance"))
    slo = _mapping_from_value(runtime_result.get("slo"))
    terrestrial_baseline = _is_terrestrial_policy(str(scheduler.get("policy", "")))
    pd_scope, pairing_rule = (
        (None, None)
        if terrestrial_baseline
        else _pd_scope_from_runtime_result(runtime_result, scope_candidate, catalog_preview)
    )
    pair_candidate_count = None if terrestrial_baseline else _pd_pair_candidate_count(catalog_preview)
    satellite_budget = (
        {}
        if terrestrial_baseline
        else _satellite_budget_from_matrix_inputs(
            context=context,
            runtime_result=runtime_result,
            catalog_preview=catalog_preview,
        )
    )
    catalog_warnings = (
        () if terrestrial_baseline else tuple(getattr(catalog_preview, "warnings", ()) or ())
    )
    catalog_fail_closed = (
        False
        if terrestrial_baseline
        else bool(getattr(catalog_preview, "provenance", {}).get("fail_closed"))
    )
    cost_payload = _mapping_from_value(context.cost_result)
    single_satellite_cost = _mapping_from_value(cost_payload.get("single_satellite"))
    cost_feasible = single_satellite_cost.get("feasible")
    terrestrial_cost = _mapping_from_value(
        _mapping_from_value(runtime_result.get("details")).get("terrestrial_costing")
    )
    terrestrial_cost_evaluated = bool(terrestrial_cost.get("evaluated", False))
    kv_policy = _mapping_from_value(scheduler.get("kv_policy"))
    kv_policy_id = _optional_str(scheduler.get("kv_policy_id"))
    warnings = tuple(
        dict.fromkeys(
            (
                *_string_tuple(runtime_result.get("warnings")),
                *catalog_warnings,
                *(("pd_scope_candidate_catalog_fail_closed",) if catalog_fail_closed else ()),
                *_string_tuple(scoring.get("warnings")),
            )
        )
    )
    return DSERuntimeMatrixRow(
        design_id=context.design_id,
        strategy_id=strategy_id,
        policy=str(scheduler.get("policy", "")),
        static_screen_selected=True,
        static_candidate_selected=True,
        static_screen_reason=static_screen_reason,
        static_rank=_optional_int(static_info.get("rank")),
        static_score=_optional_float(static_info.get("static_score")),
        static_frontier_member=bool(static_info.get("frontier_member", False)),
        static_served_ratio=_optional_float(static_info.get("served_ratio")),
        static_e2e_p95_s=_optional_float(static_info.get("e2e_p95_s")),
        static_ttft_p95_s=_optional_float(static_info.get("ttft_p95_s")),
        static_annualized_cost_usd_per_year=_optional_float(static_info.get("annualized_cost_usd_per_year")),
        placement_scope_strategy_id=placement_scope_strategy_id,
        pd_placement_scope=pd_scope,
        pd_pairing_rule=pairing_rule,
        pd_scope_candidate_count=scope_candidate_count,
        pd_pair_candidate_count=pair_candidate_count,
        serving_satellite_budget_count=_optional_int(satellite_budget.get("serving_satellite_budget_count")),
        satellite_budget_mode=_optional_str(satellite_budget.get("satellite_budget_mode")),
        comparison_group_id=_optional_str(satellite_budget.get("comparison_group_id")),
        prefill_pool_satellites=_optional_int(satellite_budget.get("prefill_pool_satellites")),
        decode_pool_satellites=_optional_int(satellite_budget.get("decode_pool_satellites")),
        kv_policy_id=kv_policy_id,
        kv_serve_actions_enabled=_string_tuple(kv_policy.get("kv_serve_actions_enabled")),
        kv_post_turn_actions_enabled=_string_tuple(kv_policy.get("kv_post_turn_actions_enabled")),
        replica_count_limit=_optional_int(kv_policy.get("replica_count_limit")),
        replica_target_deployment_id=_optional_str(kv_policy.get("replica_target_deployment_id")),
        kv_page_size_tokens=_optional_int(kv_policy.get("kv_page_size_tokens")),
        kv_transfer_chunk_bytes=_optional_float(kv_policy.get("kv_transfer_chunk_bytes")),
        eviction_policy=_optional_str(kv_policy.get("eviction_policy")),
        kv_hbm_model=_optional_str(kv_policy.get("hbm_model")),
        kv_transfer_model="page_chunk_store_and_forward" if kv_policy.get("kv_transfer_chunk_bytes") is not None else None,
        catalog_rebuilt=not terrestrial_baseline,
        catalog_builder=(
            None
            if terrestrial_baseline
            else "orbitserve.runtime_deployments.build_runtime_deployment_catalog"
        ),
        catalog_warnings=catalog_warnings,
        scheduler=scheduler,
        runtime_evaluated=bool(runtime_result.get("evaluated", False)) and not catalog_fail_closed,
        objective_score=(
            None
            if catalog_fail_closed or cost_feasible is False
            else _optional_float(scoring.get("objective_score"))
        ),
        selected=False,
        pareto_member=False,
        objective_score_scope=(
            None
            if catalog_fail_closed or cost_feasible is False
            else _optional_str(scoring.get("objective_score_scope"))
        ),
        valid_for_paper_main_results=(
            False
            if catalog_fail_closed or cost_feasible is False
            else bool(scoring.get("valid_for_paper_main_results", False))
        ),
        cost_total_usd=(
            _optional_float(terrestrial_cost.get("total_cost_usd"))
            if terrestrial_baseline and terrestrial_cost_evaluated
            else None
            if terrestrial_baseline
            else _optional_float(cost_payload.get("total_cost_usd"))
        ),
        annualized_cost_usd_per_year=(
            _optional_float(terrestrial_cost.get("annualized_cost_usd_per_year"))
            if terrestrial_baseline and terrestrial_cost_evaluated
            else None
            if terrestrial_baseline
            else _optional_float(cost_payload.get("annualized_cost_usd_per_year"))
        ),
        served_ratio=_optional_float(_first_present(slo.get("served_ratio"), runtime_metrics.get("served_ratio"))),
        leo_served_ratio=_optional_float(runtime_metrics.get("leo_served_ratio")),
        rejected_ratio=_optional_float(_first_present(slo.get("rejected_ratio"), runtime_metrics.get("rejected_ratio"))),
        timeout_ratio=_optional_float(_first_present(slo.get("timeout_ratio"), runtime_metrics.get("timeout_ratio"))),
        ttft_p95_s=_optional_float(
            _first_present(performance.get("ttft_p95_s"), runtime_metrics.get("ttft_p95_s"))
        ),
        ttft_p99_s=_optional_float(
            _first_present(performance.get("ttft_p99_s"), runtime_metrics.get("ttft_p99_s"))
        ),
        e2e_p95_s=_optional_float(
            _first_present(performance.get("e2e_p95_s"), runtime_metrics.get("e2e_p95_s"))
        ),
        e2e_p99_s=_optional_float(
            _first_present(performance.get("e2e_p99_s"), runtime_metrics.get("e2e_p99_s"))
        ),
        queue_delay_p95_s=_optional_float(
            _first_present(
                performance.get("queue_delay_p95_s"),
                runtime_metrics.get("prefill_queue_wait_p95_s"),
                runtime_metrics.get("decode_queue_wait_p95_s"),
            )
        ),
        link_delay_p95_s=_optional_float(
            _first_present(performance.get("link_delay_p95_s"), runtime_metrics.get("link_queue_wait_p95_s"))
        ),
        useful_request_throughput_per_s=_optional_float(runtime_metrics.get("useful_request_throughput_per_s")),
        useful_token_throughput_per_s=_optional_float(runtime_metrics.get("useful_token_throughput_per_s")),
        cost_per_served_request_usd=_optional_float(runtime_metrics.get("cost_per_served_request_usd")),
        cost_per_served_mtoken_usd=_optional_float(runtime_metrics.get("cost_per_served_mtoken_usd")),
        energy_violation_count=_optional_int(
            _first_present(slo.get("energy_violation_count"), runtime_metrics.get("energy_violation_count"))
        ),
        fallback_request_fraction=_optional_float(runtime_metrics.get("fallback_request_fraction")),
        deferred_request_fraction=_optional_float(runtime_metrics.get("deferred_request_fraction")),
        shadow_affected_request_fraction=_optional_float(runtime_metrics.get("shadow_affected_request_fraction")),
        shadow_served_ratio=_optional_float(runtime_metrics.get("shadow_served_ratio")),
        shadow_dvfs_request_fraction=_optional_float(runtime_metrics.get("shadow_dvfs_request_fraction")),
        shadow_offload_request_fraction=_optional_float(runtime_metrics.get("shadow_offload_request_fraction")),
        shadow_local_compute_request_fraction=_optional_float(runtime_metrics.get("shadow_local_compute_request_fraction")),
        cross_satellite_forward_request_fraction=_optional_float(runtime_metrics.get("cross_satellite_forward_request_fraction")),
        trace_model=_optional_str(_first_present(runtime_result.get("trace_model"), runtime_metrics.get("trace_model"))),
        scope=_optional_str(_first_present(runtime_result.get("scope"), runtime_metrics.get("scope"))),
        runtime_result=runtime_result,
        warnings=warnings,
        provenance={
            "design_id": context.design_id,
            "strategy_id": strategy_id,
            "placement_scope_strategy_id": placement_scope_strategy_id,
            "pd_placement_scope": pd_scope,
            "pd_pairing_rule": pairing_rule,
            "satellite_budget": satellite_budget,
            "kv_policy_id": kv_policy_id,
            "kv_policy": kv_policy,
            "pd_scope_candidate_count": scope_candidate_count,
            "pd_pair_candidate_count": pair_candidate_count,
            "catalog_fail_closed": catalog_fail_closed,
            "cost_feasible": cost_feasible,
            "comparison_domain": "terrestrial_reference" if terrestrial_baseline else "leo_co_design",
            "catalog_rebuilt": not terrestrial_baseline,
            "catalog_builder": (
                None
                if terrestrial_baseline
                else "orbitserve.runtime_deployments.build_runtime_deployment_catalog"
            ),
            "catalog_warnings": catalog_warnings,
            "scope_candidate": to_jsonable(scope_candidate),
            "runtime_evaluator": "orbitserve.runtime.evaluate_runtime_performance",
            "scorer": "orbitserve.dse.runtime_optimization.score_runtime_evaluator_result",
            "objective_score_scope": scoring.get("objective_score_scope"),
            "valid_for_paper_main_results": bool(scoring.get("valid_for_paper_main_results", False)),
            **_mapping_from_value(scoring.get("provenance")),
        },
    )


def _skipped_matrix_row(
    *,
    context: DSEDesignContext,
    static_info: Mapping[str, Any],
    reason: str,
) -> DSERuntimeMatrixRow:
    cost_payload = _mapping_from_value(context.cost_result)
    warnings = (f"runtime matrix evaluation skipped by static screening: {reason}",)
    return DSERuntimeMatrixRow(
        design_id=context.design_id,
        strategy_id="static_screen_skipped",
        policy="static_screen_skipped",
        static_screen_selected=False,
        static_candidate_selected=False,
        static_screen_reason=reason,
        static_rank=_optional_int(static_info.get("rank")),
        static_score=_optional_float(static_info.get("static_score")),
        static_frontier_member=bool(static_info.get("frontier_member", False)),
        static_served_ratio=_optional_float(static_info.get("served_ratio")),
        static_e2e_p95_s=_optional_float(static_info.get("e2e_p95_s")),
        static_ttft_p95_s=_optional_float(static_info.get("ttft_p95_s")),
        static_annualized_cost_usd_per_year=_optional_float(static_info.get("annualized_cost_usd_per_year")),
        placement_scope_strategy_id=None,
        pd_placement_scope=None,
        pd_pairing_rule=None,
        pd_scope_candidate_count=None,
        pd_pair_candidate_count=None,
        serving_satellite_budget_count=None,
        satellite_budget_mode=None,
        comparison_group_id=None,
        prefill_pool_satellites=None,
        decode_pool_satellites=None,
        kv_policy_id=None,
        kv_serve_actions_enabled=(),
        kv_post_turn_actions_enabled=(),
        replica_count_limit=None,
        replica_target_deployment_id=None,
        kv_page_size_tokens=None,
        kv_transfer_chunk_bytes=None,
        eviction_policy=None,
        kv_hbm_model=None,
        kv_transfer_model=None,
        catalog_rebuilt=False,
        catalog_builder=None,
        catalog_warnings=(),
        scheduler={},
        runtime_evaluated=False,
        objective_score=None,
        selected=False,
        pareto_member=False,
        objective_score_scope=None,
        valid_for_paper_main_results=False,
        cost_total_usd=_optional_float(cost_payload.get("total_cost_usd")),
        annualized_cost_usd_per_year=_optional_float(cost_payload.get("annualized_cost_usd_per_year")),
        served_ratio=None,
        leo_served_ratio=None,
        rejected_ratio=None,
        timeout_ratio=None,
        ttft_p95_s=None,
        ttft_p99_s=None,
        e2e_p95_s=None,
        e2e_p99_s=None,
        queue_delay_p95_s=None,
        link_delay_p95_s=None,
        useful_request_throughput_per_s=None,
        useful_token_throughput_per_s=None,
        cost_per_served_request_usd=None,
        cost_per_served_mtoken_usd=None,
        energy_violation_count=None,
        fallback_request_fraction=None,
        deferred_request_fraction=None,
        shadow_affected_request_fraction=None,
        shadow_served_ratio=None,
        shadow_dvfs_request_fraction=None,
        shadow_offload_request_fraction=None,
        shadow_local_compute_request_fraction=None,
        cross_satellite_forward_request_fraction=None,
        trace_model=None,
        scope="static_screening_skipped",
        runtime_result={
            "evaluated": False,
            "module_status": {"runtime_matrix": "skipped_by_static_screening"},
            "provenance": {
                "reason": reason,
                "static_rank": _optional_int(static_info.get("rank")),
                "static_frontier_member": bool(static_info.get("frontier_member", False)),
            },
            "warnings": warnings,
        },
        warnings=warnings,
        provenance={
            "design_id": context.design_id,
            "strategy_id": "static_screen_skipped",
            "runtime_evaluator": "not_called_static_screening",
            "scorer": "not_called_static_screening",
            "static_screening": {
                "selected": False,
                "reason": reason,
                "static_rank": _optional_int(static_info.get("rank")),
                "static_score": _optional_float(static_info.get("static_score")),
                "static_frontier_member": bool(static_info.get("frontier_member", False)),
            },
            "valid_for_paper_main_results": False,
        },
    )


def _satellite_budget_from_matrix_inputs(
    *,
    context: DSEDesignContext,
    runtime_result: Mapping[str, Any],
    catalog_preview: Any,
) -> dict[str, Any]:
    details = runtime_result.get("details") if isinstance(runtime_result.get("details"), Mapping) else _mapping_from_value(runtime_result.get("details"))
    runtime_catalog = details.get("deployment_catalog") if isinstance(details.get("deployment_catalog"), Mapping) else _mapping_from_value(details.get("deployment_catalog"))
    runtime_catalog_provenance = _mapping_from_value(runtime_catalog.get("provenance"))
    preview_provenance = _mapping_from_value(getattr(catalog_preview, "provenance", {}))
    performance_details = _design_row_performance_details(getattr(context, "design_row", None))
    performance_pd = _mapping_from_value(_mapping_from_value(performance_details.get("provenance")).get("pd"))
    keys = (
        "serving_satellite_budget_count",
        "constellation_satellite_count",
        "satellite_budget_mode",
        "satellite_budget_source",
        "comparison_group_id",
        "satellite_budget_unused_count",
        "role_budget_unused_count",
        "pd_role_budget_source",
        "prefill_pool_satellites",
        "decode_pool_satellites",
    )
    out: dict[str, Any] = {}
    for key in keys:
        value = _first_present(
            preview_provenance.get(key),
            runtime_catalog_provenance.get(key),
            performance_details.get(key),
            performance_pd.get(key),
        )
        if value is not None:
            out[key] = value
    return to_jsonable(out)


def _design_row_performance_details(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        performance = row.get("performance")
    else:
        performance = getattr(row, "performance", None)
    if isinstance(performance, Mapping):
        details = performance.get("details")
    else:
        details = getattr(performance, "details", None)
    return details if isinstance(details, dict) else _mapping_from_value(details)


def _pd_scope_from_runtime_result(
    runtime_result: Mapping[str, Any],
    scope_candidate: Mapping[str, Any],
    catalog_preview: Any,
) -> tuple[str | None, str | None]:
    serving = _mapping_from_value(runtime_result.get("serving"))
    details = _mapping_from_value(runtime_result.get("details"))
    catalog = _mapping_from_value(details.get("deployment_catalog")) or serving
    catalog_provenance = _mapping_from_value(catalog.get("provenance"))
    preview_provenance = _mapping_from_value(getattr(catalog_preview, "provenance", {}))
    pd_placement = _mapping_from_value(scope_candidate.get("pd_placement"))
    candidate_scope = _optional_str(pd_placement.get("placement_scope"))
    scope = (
        _optional_str(catalog_provenance.get("placement_scope"))
        or _optional_str(preview_provenance.get("placement_scope"))
        or candidate_scope
    )
    request_records = tuple(item for item in runtime_result.get("request_records", ()) or () if isinstance(item, dict))
    pairing_rule = None
    for record in request_records:
        pairing_rule = _optional_str(record.get("pd_pairing_rule")) or _optional_str(record.get("pd_placement_scope"))
        if pairing_rule is not None:
            break
    if pairing_rule is None:
        pairing_rule = _catalog_pairing_rule(catalog_preview, pd_placement)
    return scope, pairing_rule


def _catalog_pairing_rule(catalog_preview: Any, pd_placement: Mapping[str, Any]) -> str | None:
    for deployment in tuple(getattr(catalog_preview, "prefill_deployments", ()) or ()):
        rules = getattr(deployment, "paired_deployment_rules", {}) or {}
        if isinstance(rules, dict) and rules:
            return str(next(iter(rules.values())))
    placement_scope = _optional_str(pd_placement.get("placement_scope"))
    if placement_scope == "hybrid_explicit_pairs":
        return "user_explicit"
    if placement_scope == "hybrid_template_mix":
        template = pd_placement.get("hybrid_template")
        if isinstance(template, Mapping):
            template_id = _optional_str(template.get("template_id") or template.get("id") or template.get("name"))
        else:
            template_id = _optional_str(template)
        return f"hybrid_template:{template_id}" if template_id else "hybrid_template"
    return placement_scope


def _pd_pair_candidate_count(catalog_preview: Any) -> int | None:
    prefill_deployments = tuple(getattr(catalog_preview, "prefill_deployments", ()) or ())
    if not prefill_deployments:
        if getattr(catalog_preview, "provenance", {}).get("fail_closed"):
            return 0
        return None
    return sum(len(tuple(getattr(deployment, "paired_deployment_ids", ()) or ())) for deployment in prefill_deployments)


def _selected_keys_by_design(rows: list[DSERuntimeMatrixRow]) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    design_ids = tuple(dict.fromkeys(row.design_id for row in rows))
    for design_id in design_ids:
        scored = [
            row
            for row in rows
            if row.design_id == design_id
            and row.objective_score is not None
            and not _is_terrestrial_policy(row.policy)
        ]
        if scored:
            best = min(scored, key=lambda row: float(row.objective_score))
            selected.add((best.design_id, best.strategy_id))
    return selected


def _pareto_keys(rows: list[DSERuntimeMatrixRow]) -> set[tuple[str, str]]:
    scored = [
        row
        for row in rows
        if row.objective_score is not None and not _is_terrestrial_policy(row.policy)
    ]
    pareto: set[tuple[str, str]] = set()
    for row in scored:
        if not any(_dominates(other, row) for other in scored if other is not row):
            pareto.add((row.design_id, row.strategy_id))
    return pareto


def _dominates(left: DSERuntimeMatrixRow, right: DSERuntimeMatrixRow) -> bool:
    if _is_terrestrial_policy(left.policy) or _is_terrestrial_policy(right.policy):
        return False
    comparisons: list[tuple[float, float, str]] = []
    left_cost = _first_present(left.annualized_cost_usd_per_year, left.cost_total_usd)
    right_cost = _first_present(right.annualized_cost_usd_per_year, right.cost_total_usd)
    if left_cost is not None and right_cost is not None:
        comparisons.append((float(left_cost), float(right_cost), "lower"))
    if left.e2e_p95_s is not None and right.e2e_p95_s is not None:
        comparisons.append((float(left.e2e_p95_s), float(right.e2e_p95_s), "lower"))
    if left.served_ratio is not None and right.served_ratio is not None:
        comparisons.append((float(left.served_ratio), float(right.served_ratio), "higher"))
    if not comparisons:
        return False
    no_worse = all(a <= b if direction == "lower" else a >= b for a, b, direction in comparisons)
    strictly_better = any(a < b if direction == "lower" else a > b for a, b, direction in comparisons)
    return no_worse and strictly_better


def _pareto_payload(row: DSERuntimeMatrixRow) -> dict[str, Any]:
    return {
        "design_id": row.design_id,
        "strategy_id": row.strategy_id,
        "policy": row.policy,
        "static_rank": row.static_rank,
        "static_score": row.static_score,
        "static_frontier_member": row.static_frontier_member,
        "static_served_ratio": row.static_served_ratio,
        "static_e2e_p95_s": row.static_e2e_p95_s,
        "placement_scope_strategy_id": row.placement_scope_strategy_id,
        "pd_placement_scope": row.pd_placement_scope,
        "pd_pairing_rule": row.pd_pairing_rule,
        "pd_pair_candidate_count": row.pd_pair_candidate_count,
        "objective_score": row.objective_score,
        "objective_score_scope": row.objective_score_scope,
        "valid_for_paper_main_results": row.valid_for_paper_main_results,
        "annualized_cost_usd_per_year": row.annualized_cost_usd_per_year,
        "served_ratio": row.served_ratio,
        "leo_served_ratio": row.leo_served_ratio,
        "cost_per_served_request_usd": row.cost_per_served_request_usd,
        "cost_per_served_mtoken_usd": row.cost_per_served_mtoken_usd,
        "e2e_p95_s": row.e2e_p95_s,
        "e2e_p99_s": row.e2e_p99_s,
        "trace_model": row.trace_model,
        "scope": row.scope,
    }


def _matrix_warnings(
    rows: tuple[DSERuntimeMatrixRow, ...],
    *,
    allow_preliminary: bool,
    screening: Mapping[str, Any],
    non_separability: Mapping[str, Any],
) -> tuple[str, ...]:
    warnings = [
        "runtime_matrix.pareto_table is preliminary-model analysis and not the formal paper-main DSE Pareto frontier",
        "runtime matrix rows preserve valid_for_paper_main_results=false for preliminary runtime outputs",
    ]
    if allow_preliminary:
        warnings.append(
            "allow_preliminary_runtime_metrics=true permits preliminary runtime matrix scoring"
        )
    if rows and not any(row.objective_score is not None for row in rows):
        warnings.append(
            "strict runtime matrix mode left all rows unscored because preliminary model metrics are not allowed"
        )
    if any(not row.runtime_evaluated for row in rows):
        warnings.append("some runtime matrix rows were not evaluated by the runtime performance evaluator")
    if any(_is_terrestrial_policy(row.policy) for row in rows):
        warnings.append(
            "terrestrial baseline rows are reference-only and are excluded from LEO scheduler selection, Pareto membership, and non-separability ranking"
        )
    config = _mapping_from_value(screening.get("config"))
    if bool(config.get("enabled", False)):
        warnings.append("runtime matrix used static screening as coarse design selection before runtime evaluation")
    if screening.get("warnings"):
        warnings.extend(str(item) for item in screening.get("warnings", ()))
    if non_separability.get("warnings"):
        warnings.extend(str(item) for item in non_separability.get("warnings", ()))
    return tuple(dict.fromkeys(warnings))


def _compact_runtime_result(result: Any) -> dict[str, Any]:
    """Keep DSE matrix rows summary-sized instead of request-trace sized."""

    if isinstance(result, Mapping):
        payload = dict(result)
        compact = {
            key: to_jsonable(payload.get(key))
            for key in (
                "evaluated",
                "scenario",
                "strategy",
                "performance",
                "slo",
                "config",
                "runtime_metrics",
                "bottleneck_breakdown",
                "session_metrics",
                "communication_metrics",
                "energy_metrics",
                "region_metrics",
                "prediction_error",
                "non_separability",
                "trace_model",
                "scope",
                "valid_for_paper_main_results",
                "module_status",
                "provenance",
                "warnings",
                "module",
                "schema_version",
            )
            if key in payload
        }
        compact["workload"] = _compact_workload(payload.get("workload"))
        compact["constellation"] = _compact_constellation(payload.get("constellation"))
        compact["serving"] = _compact_serving(payload.get("serving"))
        compact["details"] = _compact_runtime_details(payload.get("details"))
        compact["decision_trace"] = _compact_decision_trace(payload.get("decision_trace"))
        compact["request_records"] = _compact_request_records(payload.get("request_records"))
        return compact
    if not hasattr(result, "evaluated") and hasattr(result, "to_dict") and callable(result.to_dict):
        return _compact_runtime_result(result.to_dict())

    compact = {
        key: to_jsonable(getattr(result, key))
        for key in (
            "evaluated",
            "scenario",
            "strategy",
            "performance",
            "slo",
            "config",
            "runtime_metrics",
            "bottleneck_breakdown",
            "session_metrics",
            "communication_metrics",
            "energy_metrics",
            "region_metrics",
            "prediction_error",
            "non_separability",
            "trace_model",
            "scope",
            "valid_for_paper_main_results",
            "module_status",
            "provenance",
            "warnings",
            "module",
            "schema_version",
        )
        if hasattr(result, key)
    }
    compact["workload"] = _compact_workload(getattr(result, "workload", None))
    compact["constellation"] = _compact_constellation(getattr(result, "constellation", None))
    compact["serving"] = _compact_serving(getattr(result, "serving", None))
    compact["details"] = _compact_runtime_details(getattr(result, "details", None))
    compact["decision_trace"] = _compact_decision_trace(getattr(result, "decision_trace", None))
    compact["request_records"] = _compact_request_records(getattr(result, "request_records", None))
    return compact


def _compact_runtime_details(value: Any) -> dict[str, Any]:
    details = value if isinstance(value, Mapping) else _mapping_from_value(value)
    compact = {
        key: to_jsonable(details[key])
        for key in (
            "scope",
            "trace_model",
            "valid_for_paper_main_results",
            "metrics_contract",
            "engine_provenance",
        )
        if key in details
    }
    if "request_records" in details:
        records = details.get("request_records")
        compact["request_record_count"] = len(records) if isinstance(records, (list, tuple)) else None
        compact["request_record_sample"] = _compact_request_records(records)
    if "deployment_catalog" in details:
        compact["deployment_catalog"] = _compact_serving(details.get("deployment_catalog"))
    if "terrestrial_costing" in details:
        terrestrial_costing = _mapping_from_value(details.get("terrestrial_costing"))
        # Preserve the small price contract summary used to populate the
        # terrestrial-reference row, while keeping full request records and
        # per-event accounting out of every DSE matrix cell.
        compact["terrestrial_costing"] = {
            key: to_jsonable(terrestrial_costing[key])
            for key in (
                "evaluated",
                "total_cost_usd",
                "annualized_cost_usd_per_year",
                "compute_cost_usd",
                "egress_cost_usd",
                "service_period_s",
                "provisioned_instance_hours",
                "service_work_utilization_proxy",
                "cost_per_completed_request_usd",
                "cost_per_served_request_usd",
                "cost_per_completed_token_usd",
                "cost_per_served_token_usd",
                "cost_per_served_mtoken_usd",
                "warnings",
                "provenance",
            )
            if key in terrestrial_costing
        }
    if "state_summary" in details:
        state = _mapping_from_value(details.get("state_summary"))
        compact["state_summary"] = {
            key: to_jsonable(state[key])
            for key in (
                "request_count",
                "served_count",
                "completed_count",
                "rejected_count",
                "timeout_count",
                "energy_violation_count",
            )
            if key in state
        }
    return compact


def _compact_request_records(value: Any, *, limit: int = 8) -> tuple[dict[str, Any], ...]:
    if not value:
        return ()
    result: list[dict[str, Any]] = []
    for raw_record in tuple(value)[:limit]:
        if not isinstance(raw_record, Mapping):
            continue
        record = dict(raw_record)
        result.append(
            {
                key: to_jsonable(record[key])
                for key in (
                    "request_id",
                    "session_id",
                    "turn_index",
                    "arrival_time_s",
                    "accepted",
                    "completed",
                    "reject_reason",
                    "timeout",
                    "ttft_s",
                    "e2e_s",
                    "prefill_queue_wait_s",
                    "decode_queue_wait_s",
                    "link_queue_wait_s",
                    "link_transmission_s",
                    "action_type",
                    "ground_origin_region_id",
                    "ground_serving_region_id",
                    "ground_replica_id",
                    "ground_cross_region",
                    "ground_network_delay_s",
                    "ground_load_balancer_queue_wait_s",
                    "ground_replica_pending_wait_s",
                    "ground_prefix_cache_hit_ratio",
                    "pd_placement_scope",
                    "pd_pairing_rule",
                    "kv_serve_action",
                    "kv_post_turn_action",
                    "kv_handoff_bytes",
                    "kv_handoff_visible_s",
                    "context_recompute_tokens",
                )
                if key in record
            }
        )
    return tuple(result)


def _compact_decision_trace(value: Any) -> dict[str, Any]:
    trace = value if isinstance(value, Mapping) else _mapping_from_value(value)
    records = trace.get("records")
    return {
        key: to_jsonable(trace[key])
        for key in ("trace_schema_version", "evaluated", "trace_level", "request_records_path", "warnings")
        if key in trace
    } | {
        "record_count": len(records) if isinstance(records, (list, tuple)) else None,
    }


def _compact_serving(value: Any) -> dict[str, Any]:
    serving = value if isinstance(value, Mapping) else _mapping_from_value(value)
    prefill = serving.get("prefill_deployments")
    decode = serving.get("decode_deployments")
    deployments = serving.get("deployments")
    compact = {
        key: to_jsonable(serving[key])
        for key in (
            "schema_version",
            "pd_mode",
            "deployment_mode",
            "model_name",
            "warnings",
            "provenance",
        )
        if key in serving
    }
    if isinstance(prefill, (list, tuple)):
        compact["prefill_deployment_count"] = len(prefill)
    if isinstance(decode, (list, tuple)):
        compact["decode_deployment_count"] = len(decode)
    if isinstance(deployments, (list, tuple)):
        compact["deployment_count"] = len(deployments)
    return compact


def _compact_workload(value: Any) -> dict[str, Any]:
    workload = _mapping_from_value(value)
    return {
        key: to_jsonable(workload[key])
        for key in (
            "workload_id",
            "name",
            "request_count",
            "session_count",
            "time_horizon_s",
            "manifest_path",
        )
        if key in workload
    }


def _compact_constellation(value: Any) -> dict[str, Any]:
    constellation = _mapping_from_value(value)
    catalog = constellation.get("satellite_catalog")
    compact = {
        key: to_jsonable(constellation[key])
        for key in (
            "name",
            "architecture_id",
            "altitude_km",
            "inclination_deg",
            "plane_count",
            "planes",
            "satellites_per_plane",
            "total_satellites",
        )
        if key in constellation
    }
    if isinstance(catalog, (list, tuple)):
        compact["satellite_catalog_count"] = len(catalog)
    return compact


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
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _is_terrestrial_policy(policy: Any) -> bool:
    policy_name = str(policy or "")
    if policy_name in {"ground_regional", "skywalker_like_ground"}:
        return True
    # Keep terrestrial classification in sync with the independent
    # centralized-paper-baseline engine without treating arbitrary policies as
    # ground references.
    from ..baselines.ground.global_runtime import is_centralized_ground_policy

    return is_centralized_ground_policy(policy_name)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


__all__ = [
    "DSEDesignContext",
    "DSERuntimeMatrixRow",
    "DSERuntimeMatrixResult",
    "evaluate_runtime_matrix",
    "unevaluated_runtime_matrix",
]
