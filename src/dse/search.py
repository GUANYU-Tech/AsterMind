"""Design-space exploration skeleton for OrbitServe."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

from ..common import to_jsonable
from ..orbit import ConstellationDesign, FormationCluster
from ..experiments.constellation_cost import ConstellationCostResult
from ..experiments.design_evaluation import (
    ARCHITECTURE_SURROGATE_WARNING,
    evaluate_materialized_design,
    safe_trace_path_component,
)
from .frontier import (
    DSEFrontierResult,
    evaluate_dse_frontier,
    evaluate_static_slo_frontier,
    unevaluated_dse_frontier,
)
from ..common import DSEMetrics, PerformanceMetrics, SLOMetrics
from .satellite_budget import analyze_communication_window_satellite_budget
from .runtime_matrix import (
    DSEDesignContext,
    DSERuntimeMatrixResult,
    evaluate_runtime_matrix,
    unevaluated_runtime_matrix,
)
from .static_slo import StaticSLOProxyResult, evaluate_static_slo_proxy
from ..workload import TokenDistribution, materialize_workload_bundle


LEGACY_SHELL_SWEEP_KEYS = {
    "altitude_km",
    "inclination_deg",
    "plane_count",
    "sats_per_plane",
    "min_elevation_deg",
    "phase_factor",
    "orbit_family",
}

SHELL_SWEEP_FIELD_MAP = {
    "altitude_km": "altitude_km",
    "inclination_deg": "inclination_deg",
    "plane_count": "plane_count",
    "sats_per_plane": "sats_per_plane",
    "min_elevation_deg": "min_elevation_deg",
    "phase_factor": "phase_factor",
    "orbit_family": "orbit_family",
    "planes": "plane_count",
    "clusters_per_plane": "sats_per_plane",
}

ORBITAL_SET_SWEEP_FIELD_MAP = {
    "altitude_km": "altitude_km",
    "inclination_deg": "inclination_deg",
    "plane_count": "planes",
    "sats_per_plane": "clusters_per_plane",
    "min_elevation_deg": "min_elevation_deg",
    "phase_factor": "walker_phase",
    "orbit_family": "orbit_type",
    "planes": "planes",
    "clusters_per_plane": "clusters_per_plane",
}

CLUSTER_SWEEP_FIELD_MAP = {
    "satellites_per_cluster": "satellites_per_cluster",
    "grid_shape": "grid_shape",
    "formation_pitch_m": "pitch_m",
    "pitch_m": "pitch_m",
}

SUPPORTED_SWEEP_KEYS = (
    set(SHELL_SWEEP_FIELD_MAP)
    | set(ORBITAL_SET_SWEEP_FIELD_MAP)
    | set(CLUSTER_SWEEP_FIELD_MAP)
)


def _default_runtime_matrix_objective() -> dict[str, Any]:
    return {
        "name": "cost_slo_weighted",
        "missing_metric_policy": "keep_with_warning",
        "weights": {"served": 1.0, "tail": 1.0, "cost": 0.0},
    }


@dataclass(frozen=True)
class DSERuntimeMatrixSpec:
    enabled: bool = False
    policies: tuple[str, ...] = ()
    include_current_policy: bool = True
    candidate_schedulers: tuple[dict[str, Any], ...] = ()
    ground_config: dict[str, Any] = field(default_factory=dict)
    kv_policy_candidates: tuple[dict[str, Any], ...] = ()
    placement_scope_candidates: tuple[dict[str, Any], ...] = ()
    static_screening: dict[str, Any] = field(default_factory=dict)
    closed_loop: dict[str, Any] = field(default_factory=dict)
    allow_preliminary_runtime_metrics: bool = False
    allow_idealized_runtime_metrics: bool = False
    ignore_optional_cost: bool = False
    objective: dict[str, Any] = field(default_factory=_default_runtime_matrix_objective)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DSERuntimeMatrixSpec":
        data = data or {}
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
        objective = _default_runtime_matrix_objective()
        objective.update(dict(data.get("objective") or {}))
        return cls(
            enabled=bool(data.get("enabled", False)),
            policies=_string_tuple(data.get("policies")),
            include_current_policy=bool(data.get("include_current_policy", True)),
            candidate_schedulers=tuple(
                dict(item)
                for item in (data.get("candidate_schedulers") or ())
                if isinstance(item, dict)
            ),
            ground_config=dict(data.get("ground_config") or {}),
            kv_policy_candidates=_kv_policy_candidates_from_dict(data),
            placement_scope_candidates=_placement_scope_candidates_from_dict(data),
            static_screening=dict(
                data.get("static_screening")
                or data.get("static_to_runtime")
                or data.get("static_screen")
                or {}
            ),
            closed_loop=dict(
                data.get("closed_loop")
                or data.get("static_to_runtime_closed_loop")
                or data.get("closed_loop_dse")
                or {}
            ),
            allow_preliminary_runtime_metrics=allow_preliminary,
            allow_idealized_runtime_metrics=allow_idealized,
            ignore_optional_cost=bool(data.get("ignore_optional_cost", False)),
            objective=objective,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _placement_scope_candidates_from_dict(data: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_candidates = (
        data.get("placement_scope_candidates")
        or data.get("scope_candidates")
        or data.get("pd_placement_scope_candidates")
        or data.get("pd_scope_candidates")
        or data.get("placement_scopes")
        or dict(data.get("placement_scope_search_space") or {}).get("scope_candidates")
        or ()
    )
    candidates: list[dict[str, Any]] = []
    for item in raw_candidates:
        if isinstance(item, str):
            candidates.append(
                {
                    "strategy_id": item,
                    "pd_placement": {"placement_scope": item},
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        pd_placement = dict(payload.get("pd_placement") or {})
        placement_scope = payload.get("placement_scope") or payload.get("pd_placement_scope")
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
        strategy_id = str(
            payload.get("strategy_id")
            or payload.get("placement_scope_strategy_id")
            or pd_placement.get("placement_scope")
            or f"pd_scope_{len(candidates)}"
        )
        candidates.append(
            {
                **payload,
                "strategy_id": strategy_id,
                "placement_scope_strategy_id": strategy_id,
                "pd_placement": pd_placement,
            }
        )
    return tuple(candidates)


def _kv_policy_candidates_from_dict(data: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_candidates = (
        data.get("kv_policy_candidates")
        or dict(data.get("kv_policy_search_space") or {}).get("kv_policy_candidates")
        or ()
    )
    return tuple(dict(item) for item in raw_candidates if isinstance(item, dict))


def _design_candidates_from_dict(data: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_candidates = (
        data.get("design_candidates")
        or data.get("constellation_candidates")
        or data.get("architecture_candidates")
        or dict(data.get("design_search_space") or {}).get("design_candidates")
        or dict(data.get("constellation_search_space") or {}).get("constellation_candidates")
        or ()
    )
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        candidate_id = str(
            payload.get("candidate_id")
            or payload.get("design_id")
            or payload.get("constellation_id")
            or payload.get("name")
            or f"design_candidate_{index + 1}"
        )
        payload["candidate_id"] = candidate_id
        candidates.append(payload)
    return tuple(candidates)


@dataclass(frozen=True)
class DSESpec:
    enabled: bool = False
    sweep: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    design_candidates: tuple[dict[str, Any], ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    runtime_matrix: DSERuntimeMatrixSpec = field(default_factory=DSERuntimeMatrixSpec)
    communication_window_budget: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DSESpec":
        data = data or {}
        sweep = {
            str(key): tuple(value if isinstance(value, list) else [value])
            for key, value in data.get("sweep", {}).items()
        }
        return cls(
            enabled=bool(data.get("enabled", False)),
            sweep=sweep,
            design_candidates=_design_candidates_from_dict(data),
            constraints=dict(data.get("constraints") or data.get("design_constraints") or {}),
            runtime_matrix=DSERuntimeMatrixSpec.from_dict(data.get("runtime_matrix")),
            communication_window_budget=dict(
                data.get("communication_window_budget")
                or data.get("satellite_budget_analysis")
                or data.get("communication_window_budget_analysis")
                or {}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class DSEDesignRow:
    design_id: str
    design: dict[str, Any]
    constellation_architecture: dict[str, Any]
    cost: ConstellationCostResult
    assignments: dict[str, Any]
    base_design_id: str
    digest: str
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    slo: SLOMetrics = field(default_factory=SLOMetrics)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_kind": "dse_candidate",
            "design_id": self.design_id,
            "assignments": to_jsonable(self.assignments),
            "base_design_id": self.base_design_id,
            "digest": self.digest,
            "design": self.design,
            "constellation_architecture": to_jsonable(self.constellation_architecture),
            "cost": self.cost.to_dict(),
            "performance": self.performance.to_dict(),
            "slo": self.slo.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DSEStudyResult:
    evaluated: bool = False
    design_count: int = 0
    rows: tuple[DSEDesignRow, ...] = ()
    metrics: DSEMetrics = field(default_factory=DSEMetrics)
    runtime_matrix: DSERuntimeMatrixResult = field(default_factory=unevaluated_runtime_matrix)
    frontier: DSEFrontierResult = field(default_factory=unevaluated_dse_frontier)
    static_frontier: DSEFrontierResult = field(default_factory=unevaluated_dse_frontier)
    warnings: tuple[str, ...] = ("DSE enumeration is disabled",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_kind": "dse_study",
            "evaluated": self.evaluated,
            "design_count": self.design_count,
            "rows": [row.to_dict() for row in self.rows],
            "metrics": self.metrics.to_dict(),
            "runtime_matrix": self.runtime_matrix.to_dict(),
            "frontier": self.frontier.to_dict(),
            "static_frontier": self.static_frontier.to_dict(),
            "warnings": list(self.warnings),
        }


def run_dse(
    experiment_spec: Any,
    *,
    workload_generation: Any | None = None,
    serving_workload: Any | None = None,
    workload_warnings: tuple[str, ...] = (),
) -> DSEStudyResult:
    """Enumerate simple shell sweeps for an ExperimentSpec-like object."""

    dse_spec: DSESpec = experiment_spec.dse
    if not dse_spec.enabled:
        return DSEStudyResult()

    _validate_sweep_keys(dse_spec.sweep)
    rows: list[DSEDesignRow] = []
    design_contexts: list[DSEDesignContext] = []
    resolver_warnings: list[str] = []
    base_design: ConstellationDesign = experiment_spec.constellation
    base_dir = Path(experiment_spec.base_dir) if getattr(experiment_spec, "base_dir", None) else None
    workload, workload_generation, workload_warnings = _resolve_workload_for_dse(
        experiment_spec.workload,
        base_dir=base_dir,
        workload_generation=workload_generation,
        serving_workload=serving_workload,
        workload_warnings=workload_warnings,
    )

    design_index = 0
    for candidate_index, candidate in enumerate(_iter_design_candidates(base_design, dse_spec), start=1):
        candidate_design = candidate["design"]
        candidate_id = candidate["candidate_id"]
        candidate_payload = candidate["candidate_payload"]
        candidate_warnings = tuple(candidate["warnings"])
        _validate_unscoped_sweep_for_design(
            candidate_design,
            dse_spec.sweep,
            candidate_id=candidate_id,
        )
        for assignments in _iter_assignments(dse_spec.sweep):
            design_index += 1
            materialized_assignments = deepcopy(assignments)
            base_design_id = str(candidate_design.name)
            design, sweep_warnings = _apply_design_assignments(
                candidate_design,
                materialized_assignments,
                design_index,
                candidate_id=candidate_id,
                preserve_name_when_empty=not bool(dse_spec.design_candidates),
            )
            constraint_warnings = _validate_design_constraints(
                design,
                base_design=base_design,
                constraints=dse_spec.constraints,
                candidate_id=candidate_id,
            )
            row = _evaluate_design_row(
                experiment_spec=experiment_spec,
                dse_spec=dse_spec,
                design=design,
                assignments=materialized_assignments,
                base_design_id=base_design_id,
                candidate_id=candidate_id,
                candidate_index=candidate_index,
                candidate_payload=candidate_payload,
                candidate_warnings=candidate_warnings,
                sweep_warnings=sweep_warnings,
                constraint_warnings=constraint_warnings,
                workload=workload,
                workload_generation=workload_generation,
                workload_warnings=workload_warnings,
                base_dir=base_dir,
                resolver_warnings=resolver_warnings,
            )
            rows.append(row["row"])
            design_contexts.append(row["context"])

    runtime_matrix = (
        evaluate_runtime_matrix(tuple(design_contexts), dse_spec.runtime_matrix)
        if dse_spec.runtime_matrix.enabled
        else unevaluated_runtime_matrix(
            allow_preliminary_runtime_metrics=dse_spec.runtime_matrix.allow_preliminary_runtime_metrics,
            allow_idealized_runtime_metrics=dse_spec.runtime_matrix.allow_idealized_runtime_metrics,
            ignore_optional_cost=dse_spec.runtime_matrix.ignore_optional_cost,
        )
    )
    runtime_frontier = evaluate_dse_frontier(runtime_matrix)
    static_frontier = evaluate_static_slo_frontier(rows)
    frontier = (
        runtime_frontier
        if dse_spec.runtime_matrix.enabled
        else static_frontier
        if static_frontier.evaluated
        else runtime_frontier
    )
    frontier_status_warning = (
        f"DSE top-level cost-SLO frontier is {frontier.analysis_scope} and not valid for paper main results"
        if frontier.evaluated
        else "DSE enumerated design rows but did not compute a top-level cost-SLO frontier"
    )
    warnings = tuple(
        dict.fromkeys(
            (
                *resolver_warnings,
                *workload_warnings,
                frontier_status_warning,
                (
                    "SLO metrics come from a preliminary static workload-SLO proxy"
                    if any(row.slo.evaluated for row in rows)
                    else "SLO metrics are placeholders"
                ),
                ARCHITECTURE_SURROGATE_WARNING,
                *frontier.warnings,
            )
        )
    )
    runtime_metrics = runtime_matrix.metrics
    metrics = DSEMetrics(
        evaluated=frontier.evaluated,
        pareto_frontier_evaluated=frontier.evaluated,
        non_separability_regret=runtime_metrics.non_separability_regret
        if dse_spec.runtime_matrix.enabled
        else None,
        dominated_design_count=runtime_metrics.dominated_design_count
        if dse_spec.runtime_matrix.enabled
        else frontier.dominated_candidate_count,
        rank_disagreement_count=runtime_metrics.rank_disagreement_count
        if dse_spec.runtime_matrix.enabled
        else None,
        warnings=tuple(
            dict.fromkeys(
                (
                    *(runtime_metrics.warnings if dse_spec.runtime_matrix.enabled else ()),
                    *frontier.warnings,
                    *(
                        static_frontier.warnings
                        if static_frontier.evaluated and dse_spec.runtime_matrix.enabled
                        else ()
                    ),
                )
            )
        )
        if (runtime_metrics.warnings or frontier.warnings or static_frontier.warnings)
        else ("DSE insight metrics are not implemented yet",),
    )

    return DSEStudyResult(
        evaluated=True,
        design_count=len(rows),
        rows=tuple(rows),
        metrics=metrics,
        runtime_matrix=runtime_matrix,
        frontier=frontier,
        static_frontier=static_frontier,
        warnings=warnings,
    )


def _evaluate_design_row(
    *,
    experiment_spec: Any,
    dse_spec: DSESpec,
    design: ConstellationDesign,
    assignments: dict[str, Any],
    base_design_id: str,
    candidate_id: str,
    candidate_index: int,
    candidate_payload: dict[str, Any],
    candidate_warnings: tuple[str, ...],
    sweep_warnings: tuple[str, ...],
    constraint_warnings: tuple[str, ...],
    workload: Any,
    workload_generation: Any,
    workload_warnings: tuple[str, ...],
    base_dir: Path | None,
    resolver_warnings: list[str],
) -> dict[str, Any]:
    materialized_assignments = deepcopy(assignments)
    input_warnings = tuple(
        dict.fromkeys(
            (
                *candidate_warnings,
                *sweep_warnings,
                *constraint_warnings,
                *workload_warnings,
            )
        )
    )
    input_simulation_config = _simulation_config_for_dse_design(
        getattr(experiment_spec, "simulation", None),
        design_id=design.name,
        experiment_id=str(experiment_spec.experiment_id),
    )
    evaluation_id = (
        str(experiment_spec.experiment_id)
        if candidate_id == "base" and not materialized_assignments and not dse_spec.design_candidates
        else design.name
    )
    evaluation = evaluate_materialized_design(
        experiment_spec,
        design=design,
        workload_generation=workload_generation,
        serving_workload=workload,
        evaluation_id=evaluation_id,
        simulation_config=input_simulation_config,
        input_warnings=input_warnings,
    )
    architecture_analysis = evaluation.constellation_architecture
    satellite_architecture = evaluation.satellite_architecture
    cost = evaluation.cost
    serving_throughput = evaluation.serving_throughput
    simulation_result = evaluation.simulation
    runtime_performance = evaluation.runtime_performance
    runtime_scenario = evaluation.runtime_scenario

    row_warnings = list(evaluation.warnings)
    static_slo = evaluate_static_slo_proxy(
        workload_generation=workload_generation,
        workload_spec=None,
        serving_throughput=serving_throughput,
        simulation_result=simulation_result,
        runtime_energy_envelope=evaluation.runtime_energy_envelope,
        base_dir=base_dir,
    )
    row_warnings.extend(static_slo.warnings)
    communication_window_budget_analysis = (
        analyze_communication_window_satellite_budget(
            simulation_result=simulation_result,
            workload_generation=workload_generation,
            serving_throughput=serving_throughput,
            config=dse_spec.communication_window_budget,
            base_dir=base_dir,
        )
        if dse_spec.communication_window_budget
        else None
    )
    if communication_window_budget_analysis is not None:
        row_warnings.extend(communication_window_budget_analysis.get("warnings", ()))

    performance = serving_throughput.to_performance_metrics()
    performance = _merge_static_slo_performance(performance, static_slo)
    runtime_current_strategy = runtime_performance.to_dict()
    performance = replace(
        performance,
        details={
            **performance.details,
            "runtime": runtime_current_strategy,
            "runtime_current_strategy": runtime_current_strategy,
            "planned_energy_demand": (
                evaluation.planned_energy_demand.to_dict()
                if evaluation.planned_energy_demand is not None
                else None
            ),
            "energy_alignment": evaluation.energy_alignment.to_dict(),
            **(
                {"communication_window_budget_analysis": communication_window_budget_analysis}
                if communication_window_budget_analysis is not None
                else {}
            ),
        },
        warnings=tuple(
            dict.fromkeys(
                (
                    *performance.warnings,
                    *runtime_performance.warnings,
                )
            )
        ),
    )
    design_digest = _materialized_design_digest(design)
    design_payload = deepcopy(design.to_dict())
    design_payload["architecture_analysis"] = architecture_analysis
    design_payload["dse_candidate"] = {
        "candidate_id": candidate_id,
        "candidate_index": candidate_index,
        "assignments": deepcopy(materialized_assignments),
        "base_design_id": base_design_id,
        "digest": design_digest,
        "source": candidate_payload.get("source"),
        "description": candidate_payload.get("description"),
        "orbit_family": candidate_payload.get("orbit_family"),
        "isl_strategy": candidate_payload.get("isl_strategy"),
        "tags": candidate_payload.get("tags") or (),
    }
    row = DSEDesignRow(
        design_id=design.name,
        assignments=materialized_assignments,
        base_design_id=base_design_id,
        digest=design_digest,
        design=design_payload,
        constellation_architecture=architecture_analysis,
        cost=cost,
        performance=performance,
        slo=static_slo.slo,
        warnings=tuple(dict.fromkeys(row_warnings)),
    )
    context = DSEDesignContext(
        design_id=design.name,
        design_row=row,
        runtime_scenario=runtime_scenario,
        current_scheduler=runtime_scenario.runtime_config.to_dict(),
        cost_result=cost,
        planned_energy_demand=evaluation.planned_energy_demand,
        runtime_energy_envelope=evaluation.runtime_energy_envelope,
        runtime_sunlight_summary=evaluation.runtime_sunlight_summary,
        energy_alignment_tolerance=float(
            getattr(experiment_spec.costing, "energy_alignment_tolerance", 0.05)
        ),
    )
    return {"row": row, "context": context}


def _merge_static_slo_performance(
    performance: PerformanceMetrics,
    static_slo: StaticSLOProxyResult,
) -> PerformanceMetrics:
    static_performance = static_slo.performance
    details = {
        **performance.details,
        "static_slo_proxy": static_slo.to_dict(),
    }
    if not static_slo.evaluated:
        return replace(
            performance,
            details=details,
            warnings=tuple(dict.fromkeys((*performance.warnings, *static_slo.warnings))),
        )
    return replace(
        performance,
        ttft_p95_s=static_performance.ttft_p95_s,
        ttft_p99_s=static_performance.ttft_p99_s,
        e2e_p95_s=static_performance.e2e_p95_s,
        e2e_p99_s=static_performance.e2e_p99_s,
        queue_delay_p95_s=static_performance.queue_delay_p95_s,
        link_delay_p95_s=static_performance.link_delay_p95_s,
        feasible=static_performance.feasible,
        bottleneck=static_performance.bottleneck,
        details=details,
        warnings=tuple(dict.fromkeys((*performance.warnings, *static_performance.warnings))),
    )


_WORKLOAD_CONFIGURATION_FIELDS = (
    "bundle_dir",
)


def _resolve_workload_for_dse(
    workload: Any,
    *,
    base_dir: Path | None,
    workload_generation: Any | None = None,
    serving_workload: Any | None = None,
    workload_warnings: tuple[str, ...] = (),
) -> tuple[Any, Any | None, tuple[str, ...]]:
    if workload_generation is not None:
        if serving_workload is not None:
            return (
                serving_workload,
                workload_generation,
                tuple(dict.fromkeys((*workload_warnings, "workload bundle reused from experiment runner; DSE did not rematerialize workload"))),
            )
        summary = _mapping_from_result(getattr(workload_generation, "summary", None))
        values = _representative_token_distribution(summary)
        token_distribution = TokenDistribution.from_dict(values) if values else None
        return token_distribution, workload_generation, workload_warnings
    if not _workload_generation_configured(workload):
        return None, None, ("workload bundle is not configured",)
    result = materialize_workload_bundle(workload, base_dir=base_dir)
    summary = _mapping_from_result(getattr(result, "summary", None))
    values = _representative_token_distribution(summary)
    token_distribution = TokenDistribution.from_dict(values) if values else None
    warnings = [f"workload_generation status: {getattr(result, 'status', 'evaluated')}"]
    warnings.extend(str(w) for w in getattr(result, "warnings", ()))
    if token_distribution is None:
        warnings.append("workload summary lacks complete prompt/output p50/p95; serving uses defaults")
    return token_distribution, result, tuple(dict.fromkeys(warnings))


def _workload_generation_configured(workload: Any) -> bool:
    return bool(getattr(workload, "bundle_dir", None))


def _simulation_config_for_dse_design(
    simulation_config: Any,
    *,
    design_id: str,
    experiment_id: str,
) -> Any:
    if simulation_config is None:
        return None
    trace_output_dir = getattr(simulation_config, "trace_output_dir", None)
    trace_model = getattr(simulation_config, "trace_model", "skeleton")
    if trace_model not in {"surrogate_orbit_trace", "circular_geometry_trace"}:
        return simulation_config
    safe_design_id = _trace_dir_design_component(design_id)
    if trace_output_dir:
        candidate_trace_dir = Path(str(trace_output_dir)) / safe_design_id
    else:
        candidate_trace_dir = Path("results") / "simulation_traces" / safe_trace_path_component(experiment_id) / "dse" / safe_design_id
    try:
        return replace(simulation_config, trace_output_dir=str(candidate_trace_dir))
    except TypeError:
        return simulation_config


def _trace_dir_design_component(design_id: str) -> str:
    return safe_trace_path_component(design_id)


def _validate_unscoped_sweep_for_design(
    design: ConstellationDesign,
    sweep: dict[str, tuple[Any, ...]],
    *,
    candidate_id: str,
) -> None:
    sweep_keys = sorted(set(sweep) & SUPPORTED_SWEEP_KEYS)
    orbital_sets = tuple(getattr(design, "effective_orbital_sets", ()) or ())
    if len(orbital_sets) <= 1 or not sweep_keys:
        return
    raise ValueError(
        f"DSE design candidate {candidate_id!r} has {len(orbital_sets)} orbital sets, but "
        f"unscoped orbital/cluster sweep keys {sweep_keys} would modify only the first set; "
        "orbital-set selectors are not implemented, so split this into separate DSE studies"
    )


def _materialized_design_digest(design: ConstellationDesign) -> str:
    payload = to_jsonable(design.to_dict())
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _mapping_from_result(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return to_jsonable(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = to_jsonable(value.to_dict())
        return converted if isinstance(converted, dict) else {}
    converted = to_jsonable(value)
    return converted if isinstance(converted, dict) else {}


def _warnings_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    raw_warnings = payload.get("warnings", ())
    if isinstance(raw_warnings, str):
        return (raw_warnings,)
    if raw_warnings is None:
        return ()
    return tuple(str(warning) for warning in raw_warnings)


def _representative_token_distribution(summary: dict[str, Any]) -> dict[str, int] | None:
    token_stats = summary.get("token_stats")
    if not isinstance(token_stats, dict):
        return None
    prompt = token_stats.get("prompt_tokens")
    output = token_stats.get("output_tokens")
    if not isinstance(prompt, dict) or not isinstance(output, dict):
        return None
    values = {
        "prompt_tokens_p50": _positive_int(prompt.get("p50")),
        "prompt_tokens_p95": _positive_int(prompt.get("p95")),
        "output_tokens_p50": _positive_int(output.get("p50")),
        "output_tokens_p95": _positive_int(output.get("p95")),
    }
    if any(value is None for value in values.values()):
        return None
    return {key: int(value) for key, value in values.items() if value is not None}


def _coerce_token_distribution(data: dict[str, Any]) -> dict[str, int] | None:
    values = {
        "prompt_tokens_p50": _positive_int(data.get("prompt_tokens_p50")),
        "prompt_tokens_p95": _positive_int(data.get("prompt_tokens_p95")),
        "output_tokens_p50": _positive_int(data.get("output_tokens_p50")),
        "output_tokens_p95": _positive_int(data.get("output_tokens_p95")),
    }
    if any(value is None for value in values.values()):
        return None
    return {key: int(value) for key, value in values.items() if value is not None}


def _positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _validate_sweep_keys(sweep: dict[str, tuple[Any, ...]]) -> None:
    unknown = sorted(set(sweep) - SUPPORTED_SWEEP_KEYS)
    if unknown:
        known = ", ".join(sorted(SUPPORTED_SWEEP_KEYS))
        raise ValueError(f"unsupported DSE sweep keys {unknown}; supported keys: {known}")
    empty_ranges = sorted(key for key, values in sweep.items() if not values)
    if empty_ranges:
        raise ValueError(f"DSE sweep ranges must not be empty: {empty_ranges}")


def _iter_assignments(sweep: dict[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    if not sweep:
        return [{}]
    keys = list(sweep)
    return [dict(zip(keys, values)) for values in product(*(sweep[key] for key in keys))]


def _iter_design_candidates(
    base_design: ConstellationDesign,
    dse_spec: DSESpec,
) -> tuple[dict[str, Any], ...]:
    if not dse_spec.design_candidates:
        return (
            {
                "candidate_id": "base",
                "design": base_design,
                "candidate_payload": {"candidate_id": "base", "source": "base_experiment_constellation"},
                "warnings": (),
            },
        )
    candidates: list[dict[str, Any]] = []
    for candidate in dse_spec.design_candidates:
        design, warnings = _design_from_candidate(base_design, candidate)
        candidates.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "design": design,
                "candidate_payload": candidate,
                "warnings": warnings,
            }
        )
    return tuple(candidates)


def _design_from_candidate(
    base_design: ConstellationDesign,
    candidate: dict[str, Any],
) -> tuple[ConstellationDesign, tuple[str, ...]]:
    warnings: list[str] = []
    candidate_id = str(candidate["candidate_id"])
    raw_constellation = (
        candidate.get("constellation")
        or candidate.get("design")
        or candidate.get("constellation_design")
    )
    if raw_constellation is None:
        raw_constellation = {
            key: value
            for key, value in candidate.items()
            if key in {"name", "shells", "orbital_sets", "gateway_profile", "application_profile", "safety"}
        }
    if not isinstance(raw_constellation, dict):
        raise TypeError(f"DSE design candidate {candidate_id!r} must provide a constellation/design mapping")
    base_payload = base_design.to_dict()
    payload = {
        "name": candidate.get("name") or raw_constellation.get("name") or candidate_id,
        "shells": raw_constellation.get("shells", ()),
        "orbital_sets": raw_constellation.get("orbital_sets", base_payload.get("orbital_sets", ())),
        "gateway_profile": raw_constellation.get("gateway_profile", base_payload.get("gateway_profile", {})),
        "application_profile": raw_constellation.get(
            "application_profile",
            base_payload.get("application_profile", "ChatLLM inference"),
        ),
        "safety": raw_constellation.get("safety", base_payload.get("safety", {})),
    }
    if not raw_constellation.get("shells") and not raw_constellation.get("orbital_sets"):
        warnings.append(
            f"DSE design candidate {candidate_id} inherits base orbital_sets; only metadata/top-level fields changed"
        )
    design = ConstellationDesign.from_dict(payload)
    return design, tuple(warnings)


def _validate_design_constraints(
    design: ConstellationDesign,
    *,
    base_design: ConstellationDesign,
    constraints: dict[str, Any],
    candidate_id: str,
) -> tuple[str, ...]:
    if not constraints:
        return ()
    warnings: list[str] = []
    fixed_satellite_count = constraints.get("satellite_count")
    if fixed_satellite_count is None and bool(
        constraints.get("same_satellite_count_as_base")
        or constraints.get("require_same_satellite_count")
        or constraints.get("same_physical_satellite_count")
    ):
        fixed_satellite_count = base_design.satellite_count
    if fixed_satellite_count is not None and design.satellite_count != int(fixed_satellite_count):
        raise ValueError(
            f"DSE design candidate {candidate_id!r} violates fixed satellite_count="
            f"{int(fixed_satellite_count)}; got {design.satellite_count}"
        )
    if bool(constraints.get("require_single_satellite_config_fixed", False)):
        warnings.append(
            "single-satellite hardware is fixed by ExperimentSpec.hardware/costing; "
            "DSE design candidates are constrained to orbit geometry only"
        )
    return tuple(warnings)


def _apply_design_assignments(
    design: ConstellationDesign,
    assignments: dict[str, Any],
    index: int,
    *,
    candidate_id: str = "base",
    preserve_name_when_empty: bool = False,
) -> tuple[ConstellationDesign, tuple[str, ...]]:
    # Reparse the complete design for every row so candidates cannot share
    # mutable policy dictionaries with the base design or with each other.
    design = deepcopy(design)
    if not assignments and preserve_name_when_empty:
        return design, ()

    shell_assignments = _map_assignments(assignments, SHELL_SWEEP_FIELD_MAP)
    orbital_set_assignments = _map_assignments(assignments, ORBITAL_SET_SWEEP_FIELD_MAP)
    cluster_assignments = _map_assignments(assignments, CLUSTER_SWEEP_FIELD_MAP)
    sweep_warnings: list[str] = []

    updated_shells = design.shells
    if updated_shells and shell_assignments:
        updated_shell = replace(updated_shells[0], **shell_assignments)
        updated_shells = (updated_shell, *updated_shells[1:])

    intermediate = replace(design, shells=updated_shells)
    needs_orbital_set_update = bool(intermediate.orbital_sets) or bool(
        set(assignments) - LEGACY_SHELL_SWEEP_KEYS
    )
    updated_orbital_sets = intermediate.orbital_sets
    if needs_orbital_set_update:
        source_orbital_sets = intermediate.orbital_sets or intermediate.effective_orbital_sets
        first_orbital_set = source_orbital_sets[0]
        updated_cluster, cluster_warnings = _apply_cluster_assignments(
            first_orbital_set.cluster,
            cluster_assignments,
        )
        sweep_warnings.extend(cluster_warnings)
        updated_orbital_set = replace(
            first_orbital_set,
            **orbital_set_assignments,
            cluster=updated_cluster,
        )
        updated_orbital_sets = (updated_orbital_set, *source_orbital_sets[1:])

    suffix = "_".join(f"{key}-{value}" for key, value in assignments.items()) or "base"
    candidate_suffix = _safe_design_id(candidate_id)
    return (
        replace(
            intermediate,
            name=f"{design.name}.dse{index:03d}.{candidate_suffix}.{suffix}",
            orbital_sets=updated_orbital_sets,
        ),
        tuple(dict.fromkeys(sweep_warnings)),
    )


def _map_assignments(assignments: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    return {
        target: assignments[source]
        for source, target in field_map.items()
        if source in assignments
    }


def _apply_cluster_assignments(
    cluster: FormationCluster,
    assignments: dict[str, Any],
) -> tuple[FormationCluster, tuple[str, ...]]:
    if not assignments:
        return cluster, ()
    payload = cluster.to_dict()
    payload.update(assignments)
    warnings: list[str] = []
    if "satellites_per_cluster" in assignments and "grid_shape" not in assignments:
        satellites_per_cluster = int(assignments["satellites_per_cluster"])
        if _grid_shape_product(payload["grid_shape"]) != satellites_per_cluster:
            payload["grid_shape"] = _balanced_grid_shape(satellites_per_cluster)
            warnings.append(
                "DSE inferred cluster.grid_shape from satellites_per_cluster for the lightweight orbit surrogate"
            )
    return FormationCluster.from_dict(payload), tuple(warnings)


def _safe_design_id(value: Any) -> str:
    text = str(value or "candidate").strip().lower()
    out = []
    for char in text:
        if char.isalnum() or char in {"_", "-", "."}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "candidate"


def _grid_shape_product(value: Any) -> int:
    if isinstance(value, str):
        parts = value.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError(f"grid shape must look like 'MxN', got {value!r}")
        rows, cols = parts
    else:
        rows, cols = value
    return int(rows) * int(cols)


def _balanced_grid_shape(satellite_count: int) -> tuple[int, int]:
    if satellite_count <= 0:
        raise ValueError("satellites_per_cluster must be positive")
    best_rows = 1
    for rows in range(1, int(satellite_count**0.5) + 1):
        if satellite_count % rows == 0:
            best_rows = rows
    return (best_rows, satellite_count // best_rows)
