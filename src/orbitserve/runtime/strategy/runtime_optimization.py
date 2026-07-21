"""Minimal runtime strategy optimization skeleton for OrbitServe."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ...common import to_jsonable


EvaluatorLike = Callable[[Any, Any], Any]

MODULE_NAME = "llm_serving_runtime_optimization"
OPTIMIZER_VERSION = "runtime_optimization_skeleton.v1"

DEFAULT_BASELINE_POLICIES = (
    "local_only",
    "shortest_path",
    "load_aware",
    "energy_aware",
    "orbitserve",
)

PAPER_POLICY_IDS = {
    "local_only": "leo_local_only",
    "shortest_path": "shortest_path_forwarding",
    "orbitserve": "full_orbitserve",
}

PERFORMANCE_FIELDS = (
    "request_throughput_per_s",
    "ttft_p95_s",
    "e2e_p95_s",
    "queue_delay_p95_s",
    "link_delay_p95_s",
    "bottleneck",
)

SLO_FIELDS = (
    "completion_ratio",
    "uncompleted_ratio",

    "energy_violation_count",
    "isl_congestion_p95",
    "sgl_congestion_p95",
    "cost_per_served_mtoken_usd",
)

POLICY_SHAPE_FIELDS = frozenset(
    {
        "policy",
        "policy_family",
        "enabled_actions",
        "session_actions_enabled",
        "dvfs_modes_enabled",
        "path_policy",
        "queue_policy",
        "fallback_enabled",
        "defer_enabled",
        "max_hop_count",
    }
)


@dataclass(frozen=True)
class RuntimeOptimizationObjective:
    name: str = "cost_slo_weighted"
    missing_metric_policy: str = "keep_with_warning"
    allow_preliminary_runtime_metrics: bool = False
    allow_idealized_runtime_metrics: bool = False
    ignore_optional_cost: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {"completion": 1.0, "tail": 1.0, "cost": 1.0}
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuntimeOptimizationObjective":
        if data is None:
            return cls()
        allow_idealized = bool(data.get("allow_idealized_runtime_metrics", False))
        allow_preliminary = bool(
            data.get(
                "allow_preliminary_runtime_metrics",
                data.get(
                    "allow_surrogate_runtime_metrics",
                    data.get("allow_sanity_runtime_metrics", allow_idealized),
                ),
            )
        )
        return cls(
            name=str(data.get("name", "cost_slo_weighted")),
            missing_metric_policy=str(data.get("missing_metric_policy", "keep_with_warning")),
            allow_preliminary_runtime_metrics=allow_preliminary,
            allow_idealized_runtime_metrics=allow_idealized,
            ignore_optional_cost=bool(data.get("ignore_optional_cost", False)),
            weights={str(k): float(v) for k, v in dict(data.get("weights") or {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RuntimeStrategySearchSpace:
    enabled: bool = True
    policies: tuple[str, ...] = ()
    candidate_schedulers: tuple[dict[str, Any], ...] = ()
    current_scheduler: dict[str, Any] = field(default_factory=dict)
    include_current_policy: bool = True
    baseline_policies: tuple[str, ...] = DEFAULT_BASELINE_POLICIES
    kv_policy_candidates: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuntimeStrategySearchSpace":
        if data is None:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", True)),
            policies=_string_tuple(data.get("policies")),
            candidate_schedulers=tuple(
                _mapping_from_value(item)
                for item in (data.get("candidate_schedulers") or ())
                if _mapping_from_value(item)
            ),
            current_scheduler=_mapping_from_value(data.get("current_scheduler")),
            include_current_policy=bool(data.get("include_current_policy", True)),
            baseline_policies=_string_tuple(data.get("baseline_policies")) or DEFAULT_BASELINE_POLICIES,
            kv_policy_candidates=tuple(
                _mapping_from_value(item)
                for item in (data.get("kv_policy_candidates") or ())
                if _mapping_from_value(item)
            ),
            provenance=dict(data.get("provenance") or {}),
            warnings=_string_tuple(data.get("warnings")),
        )

    @classmethod
    def default_for_scenario(cls, scenario: Any) -> "RuntimeStrategySearchSpace":
        current_scheduler = _scheduler_from_scenario(scenario)
        current_scheduler["policy"] = _canonical_policy(str(current_scheduler.get("policy", "orbitserve")))
        policies = _unique_strings((current_scheduler["policy"], *DEFAULT_BASELINE_POLICIES))
        return cls(
            policies=policies,
            current_scheduler=current_scheduler,
            provenance={
                "source": "default_runtime_strategy_search_space",
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RuntimeStrategyScoreRow:
    strategy_id: str
    scheduler: dict[str, Any]
    evaluator_result: dict[str, Any]
    objective_score: float | None = None
    objective_score_scope: str | None = None
    valid_for_paper_main_results: bool = False
    hard_gate_summary: dict[str, Any] = field(default_factory=dict)
    bottleneck_breakdown: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RuntimeOptimizationResult:
    evaluated: bool = False
    module: str = MODULE_NAME
    scenario_id: str | None = None
    candidate_count: int = 0
    selected_strategy: dict[str, Any] | None = None
    closed_loop_evaluated: bool = False
    closed_loop_strategy_id: str | None = None
    closed_loop_runtime: dict[str, Any] = field(default_factory=dict)
    initial_runtime_summary: dict[str, Any] = field(default_factory=dict)
    hardware_optimization: dict[str, Any] = field(default_factory=dict)
    objective: RuntimeOptimizationObjective = field(default_factory=RuntimeOptimizationObjective)
    score_table: tuple[RuntimeStrategyScoreRow, ...] = ()
    pareto_table: tuple[dict[str, Any], ...] = ()
    non_separability_table: tuple[dict[str, Any], ...] = ()
    scheduler_search_space: RuntimeStrategySearchSpace = field(default_factory=RuntimeStrategySearchSpace)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["objective"] = self.objective.to_dict()
        payload["score_table"] = [row.to_dict() for row in self.score_table]
        payload["scheduler_search_space"] = self.scheduler_search_space.to_dict()
        return payload


def optimize_runtime_strategy(
    scenario: Any,
    search_space: RuntimeStrategySearchSpace | Mapping[str, Any] | None = None,
    evaluator: EvaluatorLike | Any | None = None,
    objective: RuntimeOptimizationObjective | Mapping[str, Any] | None = None,
) -> RuntimeOptimizationResult:
    """Evaluate candidate runtime strategies without inventing missing SLO metrics."""

    objective_obj = (
        objective
        if isinstance(objective, RuntimeOptimizationObjective)
        else RuntimeOptimizationObjective.from_dict(objective)
    )
    search_space_obj = _coerce_search_space(search_space, scenario)
    evaluator_obj = evaluator or _default_evaluator

    if not search_space_obj.enabled:
        return RuntimeOptimizationResult(
            evaluated=False,
            scenario_id=_scenario_id(scenario),
            candidate_count=0,
            objective=objective_obj,
            scheduler_search_space=search_space_obj,
            provenance={
                "runtime_interface_schema": "orbitserve.runtime.v1",
                "performance_evaluator": None,
                "default_evaluator_used": False,
                "optimizer_version": OPTIMIZER_VERSION,
                "status": "disabled_by_search_space",
            },
            warnings=tuple(
                dict.fromkeys(
                    (
                        *search_space_obj.warnings,
                        "runtime strategy optimization was explicitly disabled; initial runtime result is retained",
                    )
                )
            ),
        )

    rows: list[RuntimeStrategyScoreRow] = []
    for scheduler in _candidate_schedulers(search_space_obj):
        strategy = _strategy_object(scheduler)
        raw_result, evaluator_warnings = _call_evaluator(evaluator_obj, scenario, strategy)
        scoring = score_runtime_evaluator_result(raw_result, objective_obj)
        warnings = tuple(
            dict.fromkeys(
                (
                    *_string_tuple(scheduler.get("warnings")),
                    *evaluator_warnings,
                    *scoring["warnings"],
                )
            )
        )
        rows.append(
            RuntimeStrategyScoreRow(
                strategy_id=str(scheduler["strategy_id"]),
                scheduler={k: v for k, v in scheduler.items() if k != "strategy_id"},
                evaluator_result=scoring["evaluator_result"],
                objective_score=scoring["objective_score"],
                objective_score_scope=scoring["objective_score_scope"],
                valid_for_paper_main_results=scoring["valid_for_paper_main_results"],
                hard_gate_summary=scoring["hard_gate_summary"],
                bottleneck_breakdown=scoring["bottleneck_breakdown"],
                provenance={
                    "optimizer_version": OPTIMIZER_VERSION,
                    "performance_evaluator": _evaluator_name(evaluator_obj),
                    **scoring["provenance"],
                },
                warnings=warnings,
            )
        )

    selected = _select_row(rows)
    result_warnings = list(search_space_obj.warnings)
    if selected is None:
        result_warnings.append(
            "runtime evaluator did not provide complete evaluated metrics; optimizer cannot claim "
            "completion_ratio, TTFT, E2E, or cost-SLO improvements"
        )
    if any(row.objective_score is None for row in rows):
        result_warnings.append(
            "missing runtime metrics remain null; optimizer did not derive SLO metrics from peak throughput"
        )
    if selected is not None and selected.valid_for_paper_main_results is False:
        result_warnings.append(
            "selected runtime strategy uses preliminary model metrics and is not valid for paper main results"
        )

    return RuntimeOptimizationResult(
        evaluated=selected is not None,
        scenario_id=_scenario_id(scenario),
        candidate_count=len(rows),
        selected_strategy=_selected_strategy_payload(selected),
        objective=objective_obj,
        score_table=tuple(rows),
        scheduler_search_space=search_space_obj,
        provenance={
            "runtime_interface_schema": "orbitserve.runtime.v1",
            "performance_evaluator": _evaluator_name(evaluator_obj) if evaluator is not None else None,
            "default_evaluator_used": evaluator is None,
            "optimizer_version": OPTIMIZER_VERSION,
        },
        warnings=tuple(dict.fromkeys(result_warnings)),
    )


def score_runtime_evaluator_result(
    evaluator_result: Any,
    objective: RuntimeOptimizationObjective | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a runtime evaluator result using the optimizer's conservative gates."""

    objective_obj = (
        objective
        if isinstance(objective, RuntimeOptimizationObjective)
        else RuntimeOptimizationObjective.from_dict(objective)
    )
    summary = _summarize_evaluator_result(evaluator_result)
    score, score_scope, valid_for_paper, score_warnings, score_provenance = _objective_score(
        summary["evaluator_result"],
        objective_obj,
    )
    return {
        "evaluator_result": summary["evaluator_result"],
        "objective_score": score,
        "objective_score_scope": score_scope,
        "valid_for_paper_main_results": valid_for_paper,
        "hard_gate_summary": summary["hard_gate_summary"],
        "bottleneck_breakdown": summary["bottleneck_breakdown"],
        "provenance": {
            **summary["provenance"],
            **score_provenance,
        },
        "warnings": tuple(dict.fromkeys((*summary["warnings"], *score_warnings))),
    }


def _coerce_search_space(
    search_space: RuntimeStrategySearchSpace | Mapping[str, Any] | None,
    scenario: Any,
) -> RuntimeStrategySearchSpace:
    if isinstance(search_space, RuntimeStrategySearchSpace):
        if not search_space.enabled:
            return search_space
        if search_space.policies or search_space.current_scheduler or search_space.candidate_schedulers:
            return search_space
        return RuntimeStrategySearchSpace.default_for_scenario(scenario)
    if isinstance(search_space, Mapping):
        parsed = RuntimeStrategySearchSpace.from_dict(search_space)
        if not parsed.enabled:
            return parsed
        if parsed.policies or parsed.current_scheduler or parsed.candidate_schedulers:
            return parsed
    if isinstance(search_space, Iterable) and not isinstance(search_space, (str, bytes)):
        candidates = tuple(
            candidate
            for candidate in (_mapping_from_value(item) for item in search_space)
            if candidate
        )
        if candidates:
            return RuntimeStrategySearchSpace(
                candidate_schedulers=candidates,
                include_current_policy=False,
                provenance={"source": "iterable_runtime_strategy_search_space"},
            )
    return RuntimeStrategySearchSpace.default_for_scenario(scenario)


def _candidate_schedulers(search_space: RuntimeStrategySearchSpace) -> tuple[dict[str, Any], ...]:
    if not search_space.enabled:
        return ()
    if search_space.candidate_schedulers:
        base_candidates: list[dict[str, Any]] = []
        for index, raw_scheduler in enumerate(search_space.candidate_schedulers):
            scheduler = dict(raw_scheduler)
            scheduler["policy"] = _canonical_policy(str(scheduler.get("policy", "orbitserve")))
            scheduler.setdefault("lookahead_orbits", 1.0)
            scheduler.setdefault("h_lookahead_s", None)
            scheduler.setdefault("max_hop_count", None)
            scheduler.setdefault("fallback_enabled", False)
            scheduler.setdefault("weights", {})
            scheduler.setdefault("normalization_refs", {})
            scheduler.setdefault("strategy_id", PAPER_POLICY_IDS.get(scheduler["policy"], scheduler["policy"]))
            if scheduler["strategy_id"] in {item["strategy_id"] for item in base_candidates}:
                scheduler["strategy_id"] = f"{scheduler['strategy_id']}_{index}"
            base_candidates.append(to_jsonable(scheduler))
        return _expand_kv_policy_candidates(base_candidates, search_space)

    current = dict(search_space.current_scheduler or {"policy": "orbitserve", "lookahead_orbits": 1.0})
    current["policy"] = _canonical_policy(str(current.get("policy", "orbitserve")))
    policies = list(search_space.policies or search_space.baseline_policies)
    if search_space.include_current_policy:
        policies.insert(0, current["policy"])

    base_candidates: list[dict[str, Any]] = []
    current_policy = current["policy"]
    for index, policy in enumerate(_unique_strings(_canonical_policy(str(policy)) for policy in policies)):
        preserve_current_shape = index == 0 and policy == current_policy and search_space.include_current_policy
        scheduler = _scheduler_for_policy(current, policy, preserve_current_shape=preserve_current_shape)
        scheduler["strategy_id"] = PAPER_POLICY_IDS.get(policy, policy)
        if policy not in _known_policies():
            scheduler["warnings"] = tuple(
                dict.fromkeys(
                    (
                        *_string_tuple(scheduler.get("warnings")),
                        f"unknown scheduler policy '{policy}' is preserved as an unevaluated candidate",
                    )
                )
            )
        base_candidates.append(to_jsonable(scheduler))
    return _expand_kv_policy_candidates(base_candidates, search_space)


def _expand_kv_policy_candidates(
    base_candidates: list[dict[str, Any]],
    search_space: RuntimeStrategySearchSpace,
) -> tuple[dict[str, Any], ...]:
    kv_candidates = _kv_policy_candidates(search_space)
    candidates: list[dict[str, Any]] = []
    for scheduler in base_candidates:
        for kv_candidate in kv_candidates:
            payload = dict(scheduler)
            base_strategy_id = str(payload["strategy_id"])
            kv_policy = _mapping_from_value(kv_candidate.get("kv_policy"))
            kv_policy_id = _optional_str(kv_candidate.get("kv_policy_id"))
            if kv_policy:
                payload["kv_policy"] = kv_policy
            if kv_policy_id is not None:
                payload["kv_policy_id"] = kv_policy_id
                payload["strategy_id"] = f"{base_strategy_id}::kv_{kv_policy_id}"
            candidates.append(to_jsonable(payload))
    return tuple(candidates)


def _kv_policy_candidates(search_space: RuntimeStrategySearchSpace) -> tuple[dict[str, Any], ...]:
    if not search_space.kv_policy_candidates:
        return ({"kv_policy_id": None, "kv_policy": None},)
    candidates: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(search_space.kv_policy_candidates):
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
        kv_policy_id = _optional_str(payload.get("kv_policy_id") or payload.get("strategy_id")) or f"kv_policy_{index}"
        candidates.append({"kv_policy_id": kv_policy_id, "kv_policy": kv_policy})
    return tuple(candidates)


def _scheduler_for_policy(
    current: Mapping[str, Any],
    policy: str,
    *,
    preserve_current_shape: bool,
) -> dict[str, Any]:
    common = {
        key: value
        for key, value in dict(current).items()
        if key not in POLICY_SHAPE_FIELDS and key != "strategy_id"
    }
    common.setdefault("lookahead_orbits", 1.0)
    common.setdefault("h_lookahead_s", None)
    common.setdefault("weights", {})
    common.setdefault("normalization_refs", {})
    defaults = _default_scheduler_for_policy(policy)
    if preserve_current_shape:
        return {**defaults, **dict(current), "policy": policy}
    return {**common, **defaults, "policy": policy}


def _default_scheduler_for_policy(policy: str) -> dict[str, Any]:
    try:
        from ..contracts import RuntimeStrategySpec

        return RuntimeStrategySpec.from_dict({"policy": policy}).to_dict()
    except Exception:
        if policy == "local_only":
            return {
                "policy": policy,
                "policy_family": "baseline",
                "enabled_actions": ("local_compute", "reject_drop"),
                "max_hop_count": 0,
                "fallback_enabled": False,
                "defer_enabled": False,
                "path_policy": "nearest",
            }
        if policy == "shortest_path":
            return {
                "policy": policy,
                "policy_family": "baseline",
                "enabled_actions": ("local_compute", "same_plane_forward", "cross_plane_forward", "reject_drop"),
                "fallback_enabled": False,
                "defer_enabled": False,
                "path_policy": "shortest_path",
            }
        if policy == "orbitserve":
            return {
                "policy": policy,
                "policy_family": "orbitserve",
                "enabled_actions": ("local_compute", "same_plane_forward", "cross_plane_forward", "defer", "reject_drop"),
                "path_policy": "k_shortest_score",
            }
        return {"policy": policy}


def _strategy_object(scheduler: Mapping[str, Any]) -> Any:
    payload = {k: v for k, v in scheduler.items() if k != "strategy_id"}
    try:
        from ..contracts import RuntimeStrategySpec

        if hasattr(RuntimeStrategySpec, "from_dict"):
            return RuntimeStrategySpec.from_dict(dict(payload))
        return RuntimeStrategySpec(**payload)
    except Exception:
        pass
    try:
        from .schedulers import SchedulerSpec

        return SchedulerSpec.from_dict(dict(payload))
    except Exception:
        return dict(payload)


def _default_evaluator(scenario: Any, strategy: Any) -> Any:
    try:
        from ..evaluation import evaluate_runtime_performance
    except Exception as exc:
        return _unevaluated_result(f"default runtime evaluator is unavailable: {exc}")
    try:
        return evaluate_runtime_performance(scenario, strategy)
    except Exception as exc:
        return _unevaluated_result(f"default runtime evaluator failed: {exc}")


def _call_evaluator(evaluator: Any, scenario: Any, strategy: Any) -> tuple[Any, tuple[str, ...]]:
    try:
        if callable(evaluator):
            return evaluator(scenario, strategy), ()
        evaluate = getattr(evaluator, "evaluate", None)
        if callable(evaluate):
            return evaluate(scenario, strategy), ()
    except Exception as exc:
        warning = f"runtime evaluator raised an exception: {exc}"
        return _unevaluated_result(warning), (warning,)
    warning = "runtime evaluator is not callable and has no evaluate method"
    return _unevaluated_result(warning), (warning,)


def _unevaluated_result(reason: str) -> dict[str, Any]:
    return {
        "evaluated": False,
        "performance": {field_name: None for field_name in PERFORMANCE_FIELDS},
        "slo": {field_name: None for field_name in SLO_FIELDS},
        "details": {},
        "hard_gate_summary": {},
        "bottleneck_breakdown": {},
        "provenance": {"reason": reason},
        "warnings": (reason,),
    }


def _summarize_evaluator_result(result: Any) -> dict[str, Any]:
    payload = _lightweight_evaluator_payload(result)
    performance = _mapping_from_value(payload.get("performance"))
    slo = _mapping_from_value(payload.get("slo"))
    runtime_metrics = _mapping_from_value(payload.get("runtime_metrics"))
    if not performance and runtime_metrics:
        performance = {
            "request_throughput_per_s": runtime_metrics.get("useful_request_throughput_per_s"),
            "ttft_p95_s": runtime_metrics.get("ttft_p95_s"),
            "e2e_p95_s": runtime_metrics.get("e2e_p95_s"),
            "queue_delay_p95_s": _first_present(
                runtime_metrics.get("prefill_queue_wait_p95_s"),
                runtime_metrics.get("decode_queue_wait_p95_s"),
            ),
            "link_delay_p95_s": runtime_metrics.get("link_queue_wait_p95_s"),
            "bottleneck": runtime_metrics.get("bottleneck"),
        }
    if not slo and runtime_metrics:
        slo = {
            "completion_ratio": runtime_metrics.get("completion_ratio"),
            "uncompleted_ratio": _first_present(
                runtime_metrics.get("uncompleted_ratio"),
                runtime_metrics.get("reject_ratio"),
            ),
            "energy_violation_count": runtime_metrics.get("energy_violation_count"),
            "isl_congestion_p95": runtime_metrics.get("isl_congestion_p95"),
            "sgl_congestion_p95": runtime_metrics.get("sgl_congestion_p95"),
            "cost_per_served_mtoken_usd": runtime_metrics.get("cost_per_served_mtoken_usd"),
        }
    details = _mapping_from_value(payload.get("details"))
    if runtime_metrics:
        scope = _first_present(runtime_metrics.get("scope"), payload.get("scope"))
        trace_model = _first_present(runtime_metrics.get("trace_model"), payload.get("trace_model"))
        valid_for_paper = _first_present(
            runtime_metrics.get("valid_for_paper_main_results"),
            payload.get("valid_for_paper_main_results"),
            details.get("valid_for_paper_main_results"),
        )
        details = {
            **details,
            "runtime_scope": scope,
            "scope": scope,
            "trace_model": trace_model,
            "valid_for_paper_main_results": valid_for_paper,
        }
    else:
        scope = _first_present(payload.get("scope"), details.get("scope"), details.get("runtime_scope"))
        trace_model = _first_present(payload.get("trace_model"), details.get("trace_model"))
        valid_for_paper = _first_present(
            payload.get("valid_for_paper_main_results"),
            details.get("valid_for_paper_main_results"),
        )
        if scope is not None or trace_model is not None or valid_for_paper is not None:
            details = {
                **details,
                "runtime_scope": scope,
                "scope": scope,
                "trace_model": trace_model,
                "valid_for_paper_main_results": valid_for_paper,
            }
    return {
        "evaluator_result": {
            "evaluated": bool(payload.get("evaluated", False)),
            "performance": {field_name: performance.get(field_name) for field_name in PERFORMANCE_FIELDS},
            "slo": {field_name: slo.get(field_name) for field_name in SLO_FIELDS},
            "details": details,
        },
        "hard_gate_summary": _mapping_from_value(
            payload.get("hard_gate_summary") or details.get("hard_gate_summary")
        ),
        "bottleneck_breakdown": _mapping_from_value(
            payload.get("bottleneck_breakdown") or details.get("bottleneck_breakdown")
        ),
        "provenance": _mapping_from_value(payload.get("provenance")),
        "warnings": _string_tuple(payload.get("warnings")),
    }


def _lightweight_evaluator_payload(result: Any) -> dict[str, Any]:
    """Read scoring fields without serializing request-level audit records."""

    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "runtime_metrics"):
        raw_details = getattr(result, "details", {}) or {}
        details = {
            key: raw_details.get(key)
            for key in (
                "scope",
                "runtime_scope",
                "trace_model",
                "valid_for_paper_main_results",
                "hard_gate_summary",
                "bottleneck_breakdown",
            )
            if key in raw_details
        }
        return {
            "evaluated": getattr(result, "evaluated", False),
            "performance": getattr(result, "performance", {}),
            "slo": getattr(result, "slo", {}),
            "runtime_metrics": getattr(result, "runtime_metrics", {}),
            "details": details,
            "hard_gate_summary": getattr(result, "hard_gate_summary", {}),
            "bottleneck_breakdown": getattr(result, "bottleneck_breakdown", {}),
            "provenance": getattr(result, "provenance", {}),
            "warnings": getattr(result, "warnings", ()),
            "scope": getattr(result, "scope", None),
            "trace_model": getattr(result, "trace_model", None),
            "valid_for_paper_main_results": getattr(
                result,
                "valid_for_paper_main_results",
                False,
            ),
        }
    return _mapping_from_value(result)


def _objective_score(
    evaluator_result: Mapping[str, Any],
    objective: RuntimeOptimizationObjective,
) -> tuple[float | None, str | None, bool, tuple[str, ...], dict[str, Any]]:
    warnings: list[str] = []
    if not evaluator_result.get("evaluated"):
        return (
            None,
            None,
            False,
            ("objective_score is null because evaluator_result.evaluated is false",),
            _score_provenance(objective, None, False),
        )
    details = _mapping_from_value(evaluator_result.get("details"))
    valid_for_paper = details.get("valid_for_paper_main_results") is True
    allow_preliminary = _effective_allow_preliminary_runtime_metrics(objective)
    if not valid_for_paper and not allow_preliminary:
        return (
            None,
            None,
            False,
            (
                "objective_score is null because runtime metrics are preliminary-model metrics; "
                "set allow_preliminary_runtime_metrics=true for preliminary analysis "
                "(legacy metric gates remain accepted as input aliases)",
            ),
            _score_provenance(objective, None, False),
        )
    score_scope = "paper_main" if valid_for_paper else _preliminary_score_scope(details)
    if score_scope != "paper_main":
        warnings.append(
            f"objective_score_scope={score_scope}; valid_for_paper_main_results=false"
        )
    performance = _mapping_from_value(evaluator_result.get("performance"))
    slo = _mapping_from_value(evaluator_result.get("slo"))
    completion_ratio = slo.get("completion_ratio")
    e2e_p95_s = performance.get("e2e_p95_s")
    if completion_ratio is None or e2e_p95_s is None:
        return (
            None,
            None,
            valid_for_paper,
            ("objective_score is null because completion_ratio or e2e_p95_s is missing",),
            _score_provenance(objective, None, valid_for_paper),
        )
    score = (
        -objective.weights.get("completion", 1.0) * float(completion_ratio)
        + objective.weights.get("tail", 1.0) * float(e2e_p95_s)
    )
    cost_weight = objective.weights.get("cost", 0.0)
    if cost_weight:
        cost = slo.get("cost_per_served_mtoken_usd")
        if cost is None:
            if not objective.ignore_optional_cost:
                return (
                    None,
                    None,
                    valid_for_paper,
                    ("objective_score is null because cost_per_served_mtoken_usd is missing",),
                    _score_provenance(objective, None, valid_for_paper),
                )
            warnings.append(
                "cost_per_served_mtoken_usd is missing and ignored because ignore_optional_cost=true"
            )
        else:
            score += cost_weight * float(cost)
    return score, score_scope, valid_for_paper, tuple(warnings), _score_provenance(
        objective,
        score_scope,
        valid_for_paper,
    )


def _select_row(rows: list[RuntimeStrategyScoreRow]) -> RuntimeStrategyScoreRow | None:
    scored = [row for row in rows if row.objective_score is not None]
    return min(scored, key=lambda row: float(row.objective_score)) if scored else None


def _selected_strategy_payload(row: RuntimeStrategyScoreRow | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        **row.scheduler,
        "strategy_id": row.strategy_id,
        "objective_score": row.objective_score,
        "objective_score_scope": row.objective_score_scope,
        "valid_for_paper_main_results": row.valid_for_paper_main_results,
        "warnings": row.warnings,
        "provenance": row.provenance,
    }


def _effective_allow_preliminary_runtime_metrics(objective: RuntimeOptimizationObjective) -> bool:
    return bool(objective.allow_preliminary_runtime_metrics or objective.allow_idealized_runtime_metrics)


def _preliminary_score_scope(details: Mapping[str, Any]) -> str:
    trace_model = str(details.get("trace_model") or "")
    scope = str(details.get("scope") or details.get("runtime_scope") or "")
    if "surrogate" in trace_model or "surrogate" in scope or trace_model == "circular_geometry_trace":
        return "preliminary_surrogate"
    if scope == "idealized_compute_only" or trace_model == "ideal_full_visibility":
        return "preliminary_idealized"
    return "preliminary_model"


def _score_provenance(
    objective: RuntimeOptimizationObjective,
    score_scope: str | None,
    valid_for_paper_main_results: bool,
) -> dict[str, Any]:
    return {
        "objective_name": objective.name,
        "objective_score_scope": score_scope,
        "valid_for_paper_main_results": valid_for_paper_main_results,
        "allow_preliminary_runtime_metrics": objective.allow_preliminary_runtime_metrics,
        "allow_idealized_runtime_metrics": objective.allow_idealized_runtime_metrics,
        "ignore_optional_cost": objective.ignore_optional_cost,
        "missing_metric_policy": objective.missing_metric_policy,
    }


def _scheduler_from_scenario(scenario: Any) -> dict[str, Any]:
    for key in ("runtime_config", "scheduler", "scheduler_spec", "strategy"):
        value = _value_from_object(scenario, key)
        mapping = _mapping_from_value(value)
        if mapping:
            return mapping
    manifest = _mapping_from_value(_value_from_object(scenario, "manifest"))
    mapping = _mapping_from_value(manifest.get("scheduler"))
    return mapping or {"policy": "orbitserve", "lookahead_orbits": 1.0}


def _scenario_id(scenario: Any) -> str | None:
    for key in ("scenario_id", "experiment_id", "design_id"):
        value = _value_from_object(scenario, key)
        if value is not None:
            return str(value)
    return None


def _canonical_policy(policy: str) -> str:
    return policy.strip().lower()


def _known_policies() -> set[str]:
    try:
        from .schedulers import KNOWN_POLICIES

        return set(KNOWN_POLICIES)
    except Exception:
        return {
            "local_only",
            "shortest_path",
            "load_aware",
            "energy_aware",
            "orbitserve",
        }


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


def _value_from_object(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _unique_strings(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _evaluator_name(evaluator: Any) -> str:
    if evaluator is _default_evaluator:
        return "orbitserve.runtime.evaluate_runtime_performance"
    return getattr(evaluator, "__qualname__", evaluator.__class__.__name__)


__all__ = [
    "RuntimeOptimizationObjective",
    "RuntimeStrategySearchSpace",
    "RuntimeStrategyScoreRow",
    "RuntimeOptimizationResult",
    "optimize_runtime_strategy",
    "score_runtime_evaluator_result",
]
