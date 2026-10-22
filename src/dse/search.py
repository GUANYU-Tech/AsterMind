"""DSE over current physical-network inputs and the shared system evaluator."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
import hashlib
import json
from typing import Any
from pathlib import Path

from orbitserve.common import to_jsonable
from orbitserve.runtime import RuntimeStrategySpec
from orbitserve.workload import materialize_workload_bundle
from orbitserve.experiments.design_evaluation import (
    evaluate_materialized_design,
    safe_trace_path_component,
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
    static_screening: dict[str, Any] = field(default_factory=dict)
    closed_loop: dict[str, Any] = field(default_factory=dict)
    allow_preliminary_runtime_metrics: bool = False
    ignore_optional_cost: bool = False
    objective: dict[str, Any] = field(default_factory=_default_runtime_matrix_objective)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DSERuntimeMatrixSpec":
        data = data or {}
        allow_preliminary = bool(
            data.get(
                "allow_preliminary_runtime_metrics",
                data.get(
                    "allow_surrogate_runtime_metrics",
                    data.get("allow_sanity_runtime_metrics", False),
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
            ignore_optional_cost=bool(data.get("ignore_optional_cost", False)),
            objective=objective,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _kv_policy_candidates_from_dict(data: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_candidates = (
        data.get("kv_policy_candidates")
        or dict(data.get("kv_policy_search_space") or {}).get("kv_policy_candidates")
        or ()
    )
    return tuple(dict(item) for item in raw_candidates if isinstance(item, dict))


def _string_tuple(value):
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(str(item) for item in value)


@dataclass(frozen=True)
class DSESpec:
    enabled: bool = False
    sweep: dict = field(default_factory=dict)
    design_candidates: tuple[dict, ...] = ()
    runtime_matrix: DSERuntimeMatrixSpec = field(default_factory=DSERuntimeMatrixSpec)
    constraints: dict = field(default_factory=dict)
    communication_window_budget: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data=None):
        data = data or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("Unsupported DSE fields: " + ", ".join(sorted(unknown)))
        return cls(
            bool(data.get("enabled")),
            deepcopy(data.get("sweep", {})),
            tuple(deepcopy(data.get("design_candidates", ()))),
            DSERuntimeMatrixSpec.from_dict(data.get("runtime_matrix")),
            deepcopy(data.get("constraints", {})),
            deepcopy(data.get("communication_window_budget", {})),
        )

    def to_dict(self):
        return to_jsonable(self)


def _set_path(payload, path, value):
    parts = path.split(".")
    target = payload
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    key = int(parts[-1]) if isinstance(target, list) else parts[-1]
    if isinstance(target, dict) and key not in target:
        raise ValueError("unknown physical design field: " + path)
    target[key] = deepcopy(value)


def physical_design_candidates(base, config):
    spec = config if isinstance(config, DSESpec) else DSESpec.from_dict(config)
    candidates = spec.design_candidates or ({"candidate_id": "base"},)
    seen = set()
    rows = []
    allowed_constraints = {
        "same_physical_satellite_count",
        "physical_satellite_count",
        "same_compute_node_count",
    }
    if set(spec.constraints) - allowed_constraints:
        raise ValueError("unsupported physical DSE constraints")

    def counts(value):
        from orbitserve.experiments.network_inputs import materialize_network_input

        nodes = materialize_network_input(value)["network"]["nodes"]
        return sum(n["role"] != "ground" for n in nodes), sum(
            n["role"] == "compute" and n.get("compute_target", True) for n in nodes
        )

    base_counts = counts(base) if spec.constraints else None
    for candidate in candidates:
        cid = str(candidate["candidate_id"])
        if cid in seen:
            raise ValueError("duplicate DSE candidate_id")
        seen.add(cid)
        design = deepcopy(candidate.get("physical_network", base))
        for path, value in candidate.get("patches", {}).items():
            _set_path(design, path, value)
        keys = sorted(spec.sweep)
        values = [
            spec.sweep[k]
            if isinstance(spec.sweep[k], (tuple, list))
            else (spec.sweep[k],)
            for k in keys
        ]
        if any(not v for v in values):
            raise ValueError("DSE sweep values cannot be empty")
        for combination in product(*values):
            concrete = deepcopy(design)
            assignments = dict(zip(keys, combination))
            for path, value in assignments.items():
                _set_path(concrete, path, value)
            if spec.constraints:
                count = counts(concrete)
                required = spec.constraints.get("physical_satellite_count")
                if required is not None and count[0] != int(required):
                    raise ValueError("candidate violates physical_satellite_count")
                if (
                    spec.constraints.get("same_physical_satellite_count")
                    and count[0] != base_counts[0]
                ):
                    raise ValueError("candidate violates same_physical_satellite_count")
                if (
                    spec.constraints.get("same_compute_node_count")
                    and count[1] != base_counts[1]
                ):
                    raise ValueError("candidate violates same_compute_node_count")
            digest = hashlib.sha256(
                json.dumps(concrete, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:12]
            rows.append(
                {
                    "candidate_id": cid,
                    "design_id": safe_trace_path_component(cid) + "-" + digest,
                    "physical_network": concrete,
                    "assignments": assignments,
                }
            )
    return tuple(rows)


def run_dse(
    experiment_spec,
    *,
    workload_generation=None,
    output_dir=None,
    progress=None,
    _workers=None,
    **unused,
):
    config = DSESpec.from_dict(experiment_spec.dse)
    if not config.enabled:
        return {"evaluated": False, "rows": []}
    workload = (
        workload_generation
        or materialize_workload_bundle(
            experiment_spec.workload, base_dir=experiment_spec.base_dir
        ).to_dict()
    )
    matrix = config.runtime_matrix
    # Legacy search controllers are not meaningful on physical traces. Reject
    # explicitly rather than silently accepting a configuration that does nothing.
    unsupported = {
        name for name in ("ground_config", "closed_loop") if getattr(matrix, name)
    }
    if matrix.allow_preliminary_runtime_metrics or matrix.ignore_optional_cost:
        unsupported.add("preliminary_runtime_or_optional_cost")
    if matrix.objective != _default_runtime_matrix_objective():
        unsupported.add("custom_objective")
    if unsupported:
        raise ValueError(
            "Physical DSE does not support: " + ", ".join(sorted(unsupported))
        )
    root = Path(output_dir or "results/dse")
    policies = []
    if (
        not config.runtime_matrix.enabled
        or config.runtime_matrix.include_current_policy
    ):
        policies.append(experiment_spec.runtime_strategy.to_dict())
    if config.runtime_matrix.enabled:
        policies.extend(deepcopy(config.runtime_matrix.candidate_schedulers))
        for routing in config.runtime_matrix.policies:
            policies.append(
                {
                    **experiment_spec.runtime_strategy.to_dict(),
                    "routing_policy": routing,
                }
            )
        if config.runtime_matrix.kv_policy_candidates:
            bases = policies or [experiment_spec.runtime_strategy.to_dict()]
            policies = [
                {**base, **kv}
                for base in bases
                for kv in config.runtime_matrix.kv_policy_candidates
            ]
    if not policies:
        raise ValueError("enabled DSE runtime matrix requires at least one strategy")
    rows = []
    for design in physical_design_candidates(experiment_spec.physical_network, config):
        for i, policy in enumerate(policies):
            strategy = RuntimeStrategySpec.from_dict(policy)
            evaluation = evaluate_materialized_design(
                experiment_spec,
                physical_network=design["physical_network"],
                workload_generation=workload,
                output_dir=root / design["design_id"],
                strategy=strategy,
                progress=progress,
                _workers=_workers,
            )
            diagnostics = {}
            if matrix.static_screening:
                from .static_slo import evaluate_static_slo_proxy

                diagnostics["geometry_screening"] = evaluate_static_slo_proxy(
                    simulation_result=evaluation.runtime_spec.environment_state,
                    workload_generation=workload,
                ).to_dict()
            if config.communication_window_budget.get("enabled"):
                from .satellite_budget import (
                    analyze_communication_window_satellite_budget_from_bundle,
                )

                diagnostics["communication_window_budget"] = (
                    analyze_communication_window_satellite_budget_from_bundle(
                        evaluation.runtime_spec.environment_state,
                        workload_generation=workload,
                        serving_throughput=evaluation.serving,
                        config=config.communication_window_budget,
                    )
                )
            summary = evaluation.runtime.request_summary
            total = summary["total_request_count"]
            served = summary["completed_request_count"] / total if total else 0.0
            p95 = evaluation.runtime.inference_metrics["e2e_s"]["p95"]
            rows.append(
                {
                    **{k: v for k, v in design.items() if k != "physical_network"},
                    "diagnostics": diagnostics,
                    "strategy_index": i,
                    "strategy": strategy.to_dict(),
                    "completed_fraction": served,
                    "e2e_p95_s": p95,
                    "total_cost_usd": evaluation.cost["total_cost_usd"],
                    "runtime": evaluation.runtime.to_dict(),
                    "selected": False,
                }
            )
    for row in rows:
        if row["e2e_p95_s"] is None:
            continue

        def score(r):
            return (
                -r["completed_fraction"],
                r["total_cost_usd"],
                r["e2e_p95_s"],
            )

        own = score(row)
        row["selected"] = not any(
            other is not row
            and other["e2e_p95_s"] is not None
            and all(a <= b for a, b in zip(score(other), own))
            and any(a < b for a, b in zip(score(other), own))
            for other in rows
        )
    return {
        "evaluated": True,
        "design_count": len({r["design_id"] for r in rows}),
        "row_count": len(rows),
        "rows": rows,
        "selected_rows": [r for r in rows if r["selected"]],
        "provenance": {
            "evaluator": "shared_physical_system_evaluator",
            "geometry": "per_satellite_sgp4",
        },
    }
