"""Scheduler policy interfaces for OrbitServe experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from ...common import to_jsonable


@dataclass(frozen=True)
class SchedulerDecision:
    action: str
    evaluated: bool
    policy_name: str
    rationale: str
    selected_candidate_id: str | None = None
    selected_action: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    score_terms: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0
    feasible_candidate_count: int = 0
    rejected_candidates: tuple[dict[str, Any], ...] = ()
    gate_results: dict[str, Any] = field(default_factory=dict)
    runtime_strategy: dict[str, Any] = field(default_factory=dict)
    optimization_support: dict[str, Any] = field(default_factory=dict)
    candidate_strategies: tuple[dict[str, Any], ...] = ()
    module_status: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class SchedulerPolicy(Protocol):
    name: str

    def select_action(self, state: Any, request: Any) -> SchedulerDecision:
        """Select a request action from the current simulator state."""

    def to_runtime_strategy(self) -> Any:
        """Return the runtime strategy that this scheduler exposes to evaluators."""

    def optimization_search_space(self) -> dict[str, Any]:
        """Return a scheduler search-space payload for runtime optimization."""


@dataclass(frozen=True)
class SchedulerSpec:
    policy: str = "orbitserve"
    lookahead_orbits: float = 1.0
    name: str | None = None
    policy_family: str | None = None
    enabled_actions: tuple[str, ...] = ()
    session_actions_enabled: tuple[str, ...] = ()
    dvfs_modes_enabled: tuple[str, ...] = ()
    h_lookahead_s: float | None = None
    max_hop_count: int | None = None
    fallback_enabled: bool = False
    defer_enabled: bool = False
    path_policy: str | None = None
    queue_policy: str | None = None
    hard_gate_mode: str = "strict"
    hard_gate_overrides: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, Any] = field(default_factory=dict)
    normalization_refs: dict[str, Any] = field(default_factory=dict)
    kv_policy: dict[str, Any] = field(default_factory=dict)
    ground_config: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    trace_level: str = "summary"
    runtime_optimization_enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SchedulerSpec":
        data = data or {}
        return cls(
            policy=_canonical_policy(str(data.get("policy", "orbitserve"))),
            lookahead_orbits=float(data.get("lookahead_orbits", 1.0)),
            name=_optional_str(data.get("name")),
            policy_family=_optional_str(data.get("policy_family")),
            enabled_actions=_string_tuple(data.get("enabled_actions")),
            session_actions_enabled=_string_tuple(data.get("session_actions_enabled")),
            dvfs_modes_enabled=_string_tuple(data.get("dvfs_modes_enabled")),
            h_lookahead_s=_optional_float(data.get("h_lookahead_s", data.get("H_lookahead"))),
            max_hop_count=_optional_int(data.get("max_hop_count")),
            fallback_enabled=bool(data.get("fallback_enabled", False)),
            defer_enabled=bool(data.get("defer_enabled", False)),
            path_policy=_optional_str(data.get("path_policy")),
            queue_policy=_optional_str(data.get("queue_policy")),
            hard_gate_mode=str(data.get("hard_gate_mode", "strict")),
            hard_gate_overrides=dict(data.get("hard_gate_overrides") or {}),
            weights=dict(data.get("weights") or {}),
            normalization_refs=dict(data.get("normalization_refs") or {}),
            kv_policy=dict(data.get("kv_policy") or {}),
            ground_config=dict(data.get("ground_config") or {}),
            random_seed=_optional_int(data.get("random_seed")),
            trace_level=str(data.get("trace_level", "summary")),
            runtime_optimization_enabled=bool(
                data.get("runtime_optimization_enabled", data.get("optimization_enabled", True))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "policy": self.policy,
            "lookahead_orbits": self.lookahead_orbits,
        }
        _set_if_not_none(payload, "name", self.name)
        _set_if_not_none(payload, "policy_family", self.policy_family)
        _set_if_not_empty(payload, "enabled_actions", self.enabled_actions)
        _set_if_not_empty(payload, "session_actions_enabled", self.session_actions_enabled)
        _set_if_not_empty(payload, "dvfs_modes_enabled", self.dvfs_modes_enabled)
        _set_if_not_none(payload, "h_lookahead_s", self.h_lookahead_s)
        _set_if_not_none(payload, "max_hop_count", self.max_hop_count)
        if self.fallback_enabled:
            payload["fallback_enabled"] = self.fallback_enabled
        if self.defer_enabled:
            payload["defer_enabled"] = self.defer_enabled
        _set_if_not_none(payload, "path_policy", self.path_policy)
        _set_if_not_none(payload, "queue_policy", self.queue_policy)
        if self.hard_gate_mode != "strict":
            payload["hard_gate_mode"] = self.hard_gate_mode
        _set_if_not_empty(payload, "hard_gate_overrides", self.hard_gate_overrides)
        _set_if_not_empty(payload, "weights", self.weights)
        _set_if_not_empty(payload, "normalization_refs", self.normalization_refs)
        _set_if_not_empty(payload, "kv_policy", self.kv_policy)
        _set_if_not_empty(payload, "ground_config", self.ground_config)
        _set_if_not_none(payload, "random_seed", self.random_seed)
        if self.trace_level != "summary":
            payload["trace_level"] = self.trace_level
        if not self.runtime_optimization_enabled:
            payload["runtime_optimization_enabled"] = False
        return to_jsonable(payload)


@dataclass(frozen=True)
class SchedulerCandidate:
    candidate_id: str
    action_type: str
    feasible: bool
    route_plan: dict[str, Any]
    predicted_ttft_s: float | None
    predicted_e2e_s: float | None
    predicted_queue_wait_s: float | None
    predicted_link_wait_s: float | None
    predicted_energy_risk: float | None
    predicted_soc_after: float | None
    hop_count: int
    hard_gate_violations: tuple[str, ...] = ()
    hard_gate_results: dict[str, Any] = field(default_factory=dict)
    predicted_nominal_latency_s: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "SchedulerCandidate":
        if isinstance(value, cls):
            return value
        payload = _mapping_from_value(value)
        route_plan = _mapping_from_value(payload.get("route_plan"))
        return cls(
            candidate_id=str(payload["candidate_id"]),
            action_type=str(payload.get("action_type") or "candidate"),
            feasible=bool(payload.get("feasible", True)),
            route_plan=route_plan,
            predicted_ttft_s=_optional_float(payload.get("predicted_ttft_s")),
            predicted_e2e_s=_optional_float(payload.get("predicted_e2e_s")),
            predicted_queue_wait_s=_optional_float(payload.get("predicted_queue_wait_s")),
            predicted_link_wait_s=_optional_float(payload.get("predicted_link_wait_s")),
            predicted_energy_risk=_optional_float(payload.get("predicted_energy_risk")),
            predicted_soc_after=_optional_float(payload.get("predicted_soc_after")),
            hop_count=_optional_int(payload.get("hop_count")) or 0,
            hard_gate_violations=_string_tuple(payload.get("hard_gate_violations")),
            hard_gate_results=_mapping_from_value(payload.get("hard_gate_results")),
            predicted_nominal_latency_s=_optional_float(payload.get("predicted_nominal_latency_s")),
            provenance=_mapping_from_value(payload.get("provenance")),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


RUNTIME_OPTIMIZATION_BASELINE_POLICIES = (
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


@dataclass(frozen=True)
class RuntimeScheduler:
    """Runtime-facing scheduler adapter.

    This class exposes RuntimeStrategySpec payloads for evaluation/optimization
    previews and selects among request-level candidates when runtime_engine
    provides them.
    """

    spec: SchedulerSpec

    @property
    def name(self) -> str:
        return self.spec.name or self.spec.policy

    @property
    def policy_name(self) -> str:
        return self.spec.policy

    @property
    def policy(self) -> str:
        return self.spec.policy

    def to_runtime_strategy(self) -> Any:
        from ..contracts import RuntimeStrategySpec

        strategy = RuntimeStrategySpec.from_value(self.spec)
        return RuntimeStrategySpec.from_dict(
            {
                **strategy.to_dict(),
                "provenance": {
                    **strategy.provenance,
                    "scheduler_bridge": "RuntimeScheduler",
                    "scheduler_module": "orbitserve.schedulers",
                },
            }
        )

    def runtime_strategy_payload(self) -> dict[str, Any]:
        return self.to_runtime_strategy().to_dict()

    def candidate_strategy_payloads(self, policies: tuple[str, ...] | None = None) -> tuple[dict[str, Any], ...]:
        from ..contracts import RuntimeStrategySpec

        ordered_policies = _unique_policies(
            (
                self.spec.policy,
                *(policies or RUNTIME_OPTIMIZATION_BASELINE_POLICIES),
            )
        )
        candidates: list[dict[str, Any]] = []
        for policy in ordered_policies:
            if policy not in KNOWN_POLICIES:
                known = ", ".join(sorted(KNOWN_POLICIES))
                raise ValueError(f"unknown scheduler policy '{policy}'. Known policies: {known}")
            payload = (
                self.runtime_strategy_payload()
                if policy == self.spec.policy
                else RuntimeStrategySpec.from_dict({"policy": policy}).to_dict()
            )
            payload = {
                **payload,
                "strategy_id": PAPER_POLICY_IDS.get(policy, policy),
                "provenance": {
                    **dict(payload.get("provenance") or {}),
                    "scheduler_bridge": "RuntimeScheduler",
                    "scheduler_module": "orbitserve.schedulers",
                    "scheduler_source_policy": self.spec.policy,
                },
            }
            candidates.append(to_jsonable(payload))
        return tuple(candidates)

    def optimization_search_space(self) -> dict[str, Any]:
        if not self.spec.runtime_optimization_enabled:
            return {
                "enabled": False,
                "current_scheduler": self.runtime_strategy_payload(),
                "candidate_schedulers": (),
                "include_current_policy": False,
                "baseline_policies": (),
                "warnings": ("scheduler.runtime_optimization_enabled=false",),
                "provenance": {
                    "source": "scheduler_runtime_bridge",
                    "scheduler_policy": self.spec.policy,
                    "scheduler_module": "orbitserve.schedulers",
                    "runtime_optimization_enabled": False,
                },
            }
        candidates = self.candidate_strategy_payloads()
        return {
            "current_scheduler": self.runtime_strategy_payload(),
            "candidate_schedulers": candidates,
            "include_current_policy": False,
            "baseline_policies": RUNTIME_OPTIMIZATION_BASELINE_POLICIES,
            "provenance": {
                "source": "scheduler_runtime_bridge",
                "scheduler_policy": self.spec.policy,
                "scheduler_module": "orbitserve.schedulers",
                "candidate_strategy_count": len(candidates),
            },
        }

    def runtime_search_space(self) -> dict[str, Any]:
        return self.optimization_search_space()

    def select_action(self, state: Any, request: Any) -> SchedulerDecision:
        runtime_strategy = self.runtime_strategy_payload()
        candidate_strategies = self.candidate_strategy_payloads()
        request_payload = _mapping_from_value(request)
        state_payload = _mapping_from_value(state)
        candidates = tuple(
            SchedulerCandidate.from_value(candidate)
            for candidate in state_payload.get("candidates", ())
        )
        support = {
            "runtime_strategy_available": True,
            "runtime_performance_entrypoint": "orbitserve.runtime.evaluate_runtime_performance",
            "runtime_optimization_entrypoint": "orbitserve.runtime_optimization.optimize_runtime_strategy",
            "candidate_search_space_available": True,
            "candidate_strategy_count": len(candidate_strategies),
            "candidate_policies": tuple(candidate["policy"] for candidate in candidate_strategies),
            "request_level_execution": "request_level_resource_aware_surrogate",
            "paper_rule_status": "request_level_resource_aware_surrogate",
        }
        if not candidates:
            return SchedulerDecision(
                action="scheduler_bridge/runtime_strategy_preview",
                evaluated=False,
                policy_name=self.name,
                rationale=(
                    f"{self.name} scheduler exposes a RuntimeStrategySpec bridge for "
                    "runtime performance evaluation and runtime optimization; "
                    "request-level action selection requires runtime_engine state/request"
                ),
                runtime_strategy=runtime_strategy,
                optimization_support=support,
                candidate_strategies=candidate_strategies,
                module_status={
                    "scheduling": "runtime_bridge",
                    "runtime_strategy": "provided",
                    "runtime_optimization": "candidate_strategy_payloads_available",
                },
                provenance={
                    "scheduler_interface": "runtime_bridge",
                    "state_request_available": False,
                },
            )
        hard_gate_mode = _hard_gate_mode(runtime_strategy.get("hard_gate_mode", self.spec.hard_gate_mode))
        hard_gate_blocking = _hard_gate_blocks_selection(hard_gate_mode)
        candidates = tuple(
            _with_scheduler_gate_results(
                candidate,
                request_payload,
                state_payload,
                runtime_strategy,
            )
            for candidate in candidates
        )
        normalization_refs = _normalization_refs(candidates, request_payload, state_payload, runtime_strategy)
        scored = [
            self._score_candidate(
                candidate,
                request_payload,
                runtime_strategy,
                normalization_refs,
            )
            for candidate in candidates
        ]
        feasible_scored = [
            item
            for item in scored
            if item[1].feasible and (not hard_gate_blocking or not item[1].hard_gate_violations)
        ]
        gate_results = _gate_results(candidates, hard_gate_mode=hard_gate_mode, hard_gate_blocking=hard_gate_blocking)
        if not feasible_scored:
            best_score, best_candidate, best_terms = min(
                scored,
                key=lambda item: (
                    item[0],
                    str(item[1].provenance.get("compute_satellite_id", "")),
                    str(item[1].provenance.get("access_satellite_id", "")),
                    str(item[1].provenance.get("path_id", item[1].candidate_id)),
                ),
            )
            return SchedulerDecision(
                action="reject_drop",
                evaluated=True,
                policy_name=self.name,
                rationale=(
                    f"{self.name} scheduler found no feasible request-level candidate "
                    "after hard-gate checks"
                ),
                selected_candidate_id=best_candidate.candidate_id,
                selected_action={
                    "candidate_id": best_candidate.candidate_id,
                    "action_type": best_candidate.action_type,
                    "route_plan": best_candidate.route_plan,
                },
                score=best_score,
                score_terms=best_terms,
                candidate_count=len(candidates),
                feasible_candidate_count=0,
                rejected_candidates=_rejected_candidate_payloads(
                    scored,
                    selected_candidate_id=None,
                    hard_gate_blocking=hard_gate_blocking,
                ),
                gate_results=gate_results,
                runtime_strategy=runtime_strategy,
                optimization_support=support,
                candidate_strategies=candidate_strategies,
                module_status={
                    "scheduling": "runtime_request_level",
                    "runtime_strategy": "provided",
                    "runtime_optimization": "candidate_strategy_payloads_available",
                },
                provenance={
                    "scheduler_interface": "runtime_request_level",
                    "state_request_available": True,
                    "score_model": "request_level_slo_aware_surrogate",
                    "reject_reason": "no_feasible_scheduler_candidate",
                    "hard_gate_mode": hard_gate_mode,
                },
            )
        score, selected, score_terms = min(
            feasible_scored,
            key=lambda item: (
                item[0],
                str(item[1].provenance.get("compute_satellite_id", "")),
                str(item[1].provenance.get("access_satellite_id", "")),
                str(item[1].provenance.get("path_id", item[1].candidate_id)),
            ),
        )
        return SchedulerDecision(
            action=selected.action_type,
            evaluated=True,
            policy_name=self.name,
            rationale=(
                f"{self.name} scheduler selected a request-level candidate with "
                "SLO-aware surrogate scoring"
            ),
            selected_candidate_id=selected.candidate_id,
            selected_action=selected.to_dict(),
            score=score,
            score_terms=score_terms,
            candidate_count=len(candidates),
            feasible_candidate_count=len(feasible_scored),
            rejected_candidates=_rejected_candidate_payloads(
                scored,
                selected_candidate_id=selected.candidate_id,
                hard_gate_blocking=hard_gate_blocking,
            ),
            gate_results=gate_results,
            runtime_strategy=runtime_strategy,
            optimization_support=support,
            candidate_strategies=candidate_strategies,
            module_status={
                "scheduling": "request_level_resource_aware_surrogate",
                "runtime_strategy": "provided",
                "runtime_optimization": "candidate_strategy_payloads_available",
            },
            provenance={
                "scheduler_interface": "request_level_candidate_selector",
                "state_request_available": True,
                "score_model": "request_level_resource_aware_surrogate",
                "hard_gate_mode": hard_gate_mode,
            },
        )

    def _score_candidate(
        self,
        candidate: SchedulerCandidate,
        request: dict[str, Any],
        runtime_strategy: dict[str, Any],
        refs: dict[str, float],
    ) -> tuple[float, SchedulerCandidate, dict[str, Any]]:
        weights = self._effective_policy_weights(runtime_strategy)
        hard_gate_mode = _hard_gate_mode(runtime_strategy.get("hard_gate_mode", self.spec.hard_gate_mode))
        hard_gate_blocking = _hard_gate_blocks_selection(hard_gate_mode)
        e2e = _finite_or(candidate.predicted_e2e_s, refs["latency_s"])
        ttft = _finite_or(candidate.predicted_ttft_s, e2e)
        queue = _finite_or(candidate.predicted_queue_wait_s, 0.0)
        link = _finite_or(candidate.predicted_link_wait_s, 0.0)
        energy_risk = _finite_or(candidate.predicted_energy_risk, 0.0)
        latency_for_score = _finite_or(candidate.predicted_nominal_latency_s, e2e)
        # Hard gates are enforced before ranking in select_action().  Keep this
        # compatibility field at zero so a penalty can never substitute for
        # admission filtering.
        hard_gate_penalty = 0.0
        normalized_e2e = latency_for_score / refs["latency_s"]
        normalized_queue = queue / refs["queue_s"]
        normalized_link = link / refs["link_s"]
        normalized_energy_risk = energy_risk / refs["energy_risk"]
        hop_penalty = float(candidate.hop_count)
        provenance = _mapping_from_value(candidate.provenance)
        kv_hit_tokens = _finite_or(provenance.get("kv_hit_tokens"), 0.0)
        kv_migration_visible_s = _finite_or(provenance.get("kv_migration_visible_s"), 0.0)
        kv_recompute_tokens = _finite_or(provenance.get("context_recompute_tokens"), 0.0)
        kv_hbm_occupancy = _finite_or(provenance.get("hbm_occupancy_after"), 0.0)
        compute_sunlit = provenance.get("compute_sunlit")
        access_sunlit = provenance.get("access_sunlit")
        shadow_triggered = bool(provenance.get("shadow_triggered", False))
        shadow_strategies = {
            str(item) for item in tuple(provenance.get("shadow_strategies", ()) or ())
        }
        shadow_local_penalty = (
            1.0
            if compute_sunlit is False and candidate.action_type in {"local_compute", "shadow_local_compute"}
            else 0.0
        )
        shadow_dvfs_indicator = (
            1.0
            if candidate.action_type == "shadow_dvfs" or "shadow_dvfs" in shadow_strategies
            else 0.0
        )
        shadow_offload_indicator = (
            1.0
            if candidate.action_type == "shadow_offload" or "shadow_offload" in shadow_strategies
            else 0.0
        )
        normalized_kv_migration = kv_migration_visible_s / refs["link_s"]
        normalized_kv_recompute = kv_recompute_tokens / refs["tokens"]
        normalized_kv_hbm = kv_hbm_occupancy / refs["hbm_occupancy"]
        normalized_kv_hit_credit = kv_hit_tokens / refs["tokens"]
        weighted_components = {
            "latency": _weight(weights, "latency", 1.0) * normalized_e2e,
            "queue": _weight(weights, "queue", 1.0) * normalized_queue,
            "link": _weight(weights, "link", 1.0) * normalized_link,
            "energy": _weight(weights, "energy", 10.0) * normalized_energy_risk,
            "hop": _weight(weights, "hop", 0.0) * hop_penalty,
            "kv_link": _weight(weights, "kv_link", _weight(weights, "kv", 0.0))
            * normalized_kv_migration,
            "kv_recompute": _weight(
                weights, "kv_recompute", _weight(weights, "session", 0.0)
            )
            * normalized_kv_recompute,
            "kv_hbm": _weight(weights, "kv_hbm", 0.0) * normalized_kv_hbm,
            "kv_hit": -_weight(weights, "kv_hit", 0.0) * normalized_kv_hit_credit,
            "shadow_local": _weight(weights, "shadow_local", 0.0) * shadow_local_penalty,
            "shadow_dvfs_credit": -_weight(weights, "shadow_dvfs_credit", 0.0) * shadow_dvfs_indicator,
            "shadow_offload_credit": -_weight(weights, "shadow_offload_credit", 0.0)
            * shadow_offload_indicator,
        }
        score = hard_gate_penalty + sum(weighted_components.values())
        score_components = {
            "latency": _score_component(
                latency_for_score,
                refs["latency_s"],
                normalized_e2e,
                weights,
                "latency",
                1.0,
                weighted_components,
            ),
            "queue": _score_component(
                queue, refs["queue_s"], normalized_queue, weights, "queue", 1.0, weighted_components
            ),
            "link": _score_component(
                link, refs["link_s"], normalized_link, weights, "link", 1.0, weighted_components
            ),
            "energy": _score_component(
                energy_risk,
                refs["energy_risk"],
                normalized_energy_risk,
                weights,
                "energy",
                10.0,
                weighted_components,
            ),
            "hop": _score_component(
                candidate.hop_count, 1.0, hop_penalty, weights, "hop", 0.0, weighted_components
            ),
            "kv_link": _score_component(
                kv_migration_visible_s,
                refs["link_s"],
                normalized_kv_migration,
                weights,
                "kv_link",
                _weight(weights, "kv", 0.0),
                weighted_components,
            ),
            "kv_recompute": _score_component(
                kv_recompute_tokens,
                refs["tokens"],
                normalized_kv_recompute,
                weights,
                "kv_recompute",
                _weight(weights, "session", 0.0),
                weighted_components,
            ),
            "kv_hbm": _score_component(
                kv_hbm_occupancy,
                refs["hbm_occupancy"],
                normalized_kv_hbm,
                weights,
                "kv_hbm",
                0.0,
                weighted_components,
            ),
            "kv_hit": _score_component(
                kv_hit_tokens,
                refs["tokens"],
                normalized_kv_hit_credit,
                weights,
                "kv_hit",
                0.0,
                weighted_components,
            ),
            "shadow_local": _score_component(
                shadow_local_penalty,
                1.0,
                shadow_local_penalty,
                weights,
                "shadow_local",
                0.0,
                weighted_components,
            ),
            "shadow_dvfs_credit": _score_component(
                shadow_dvfs_indicator,
                1.0,
                shadow_dvfs_indicator,
                weights,
                "shadow_dvfs_credit",
                0.0,
                weighted_components,
            ),
            "shadow_offload_credit": _score_component(
                shadow_offload_indicator,
                1.0,
                shadow_offload_indicator,
                weights,
                "shadow_offload_credit",
                0.0,
                weighted_components,
            ),
        }
        score_terms = {
            "predicted_ttft_s": candidate.predicted_ttft_s,
            "predicted_e2e_s": candidate.predicted_e2e_s,
            "predicted_queue_wait_s": candidate.predicted_queue_wait_s,
            "predicted_link_wait_s": candidate.predicted_link_wait_s,
            "predicted_energy_risk": candidate.predicted_energy_risk,
            "predicted_soc_after": candidate.predicted_soc_after,
            "hard_gate_mode": hard_gate_mode,
            "hard_gate_blocking": hard_gate_blocking,
            "hard_gate_violations": candidate.hard_gate_violations,
            "hard_gate_results": candidate.hard_gate_results,
            "hard_gate_penalty": hard_gate_penalty,
            "normalized_e2e": normalized_e2e,
            "normalized_queue": normalized_queue,
            "normalized_link": normalized_link,
            "normalized_energy_risk": normalized_energy_risk,
            "hop_penalty": hop_penalty,
            "kv_hit_tokens": kv_hit_tokens,
            "kv_migration_visible_s": kv_migration_visible_s,
            "context_recompute_tokens": kv_recompute_tokens,
            "hbm_occupancy_after": kv_hbm_occupancy,
            "compute_sunlit": compute_sunlit,
            "access_sunlit": access_sunlit,
            "shadow_triggered": shadow_triggered,
            "shadow_strategy": provenance.get("shadow_strategy"),
            "shadow_local_penalty": shadow_local_penalty,
            "shadow_dvfs_indicator": shadow_dvfs_indicator,
            "shadow_offload_indicator": shadow_offload_indicator,
            "normalized_kv_migration": normalized_kv_migration,
            "normalized_kv_recompute": normalized_kv_recompute,
            "normalized_kv_hbm": normalized_kv_hbm,
            "normalized_kv_hit_credit": normalized_kv_hit_credit,
            "weights": weights,
            "normalization_refs": refs,
            "normalization_scope": "request_candidate_set",
            "score_components": score_components,
            "score_model": "request_level_resource_aware_surrogate",
        }
        return score, candidate, to_jsonable(score_terms)

    def _effective_policy_weights(self, runtime_strategy: dict[str, Any]) -> dict[str, float]:
        policy = str(runtime_strategy.get("policy") or self.spec.policy)
        if policy in {"local_only", "nearest_visible"}:
            defaults = {"latency": 1.0, "queue": 1.0, "link": 1.0, "energy": 10.0, "hop": 0.0}
        elif policy == "shortest_path":
            defaults = {"latency": 1.0, "queue": 0.0, "link": 0.0, "energy": 0.0, "hop": 0.2}
        elif policy == "load_aware":
            defaults = {"latency": 0.25, "queue": 5.0, "link": 0.5, "energy": 0.0, "hop": 0.05}
        elif policy == "energy_aware":
            defaults = {"latency": 0.25, "queue": 0.5, "link": 0.5, "energy": 50.0, "hop": 0.05}
        else:
            defaults = {"latency": 1.0, "queue": 2.0, "link": 1.0, "energy": 10.0, "hop": 0.1}
        overrides = _mapping_from_value(runtime_strategy.get("weights"))
        for key, value in overrides.items():
            if value is not None:
                defaults[str(key)] = _weight(overrides, str(key), defaults.get(str(key), 0.0))
        return defaults



KNOWN_POLICIES = {
    "ground_regional",
    "skywalker_like_ground",
    "local_only",
    "nearest_visible",
    "shortest_path",
    "load_aware",
    "energy_aware",
    "session_sticky_only",
    "session_recompute_only",
    "orbitserve",
}


def get_scheduler(policy_or_spec: str | SchedulerSpec | dict[str, Any] | Any = "orbitserve") -> RuntimeScheduler:
    spec = _coerce_scheduler_spec(policy_or_spec)
    policy_name = spec.policy
    if policy_name not in KNOWN_POLICIES:
        known = ", ".join(sorted(KNOWN_POLICIES))
        raise ValueError(f"unknown scheduler policy '{policy_name}'. Known policies: {known}")
    return RuntimeScheduler(spec)


def _coerce_scheduler_spec(value: str | SchedulerSpec | dict[str, Any] | Any) -> SchedulerSpec:
    if isinstance(value, RuntimeScheduler):
        return replace(value.spec, policy=_canonical_policy(value.spec.policy))
    if isinstance(value, SchedulerSpec):
        return replace(value, policy=_canonical_policy(value.policy))
    if isinstance(value, str):
        return SchedulerSpec.from_dict({"policy": value})
    if isinstance(value, dict):
        return SchedulerSpec.from_dict(value)
    payload = {
        key: getattr(value, key)
        for key in (
            "policy",
            "lookahead_orbits",
            "name",
            "policy_family",
            "enabled_actions",
            "session_actions_enabled",
            "dvfs_modes_enabled",
            "h_lookahead_s",
            "max_hop_count",
            "fallback_enabled",
            "defer_enabled",
            "path_policy",
            "queue_policy",
            "hard_gate_mode",
            "hard_gate_overrides",
            "weights",
            "normalization_refs",
            "random_seed",
            "trace_level",
            "kv_policy",
        )
        if hasattr(value, key)
    }
    if not payload:
        raise TypeError("scheduler value must be a policy string, SchedulerSpec, mapping, or scheduler-like object")
    if "policy" not in payload:
        raise TypeError("scheduler-like object must expose a policy field")
    return SchedulerSpec.from_dict(payload)


def _canonical_policy(value: str) -> str:
    normalized = value.strip().lower()
    return normalized


def _mapping_from_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(to_jsonable(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        return dict(to_jsonable(payload)) if isinstance(payload, dict) else {}
    payload = {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }
    return dict(to_jsonable(payload)) if payload else {}


def _normalization_refs(
    candidates: tuple[SchedulerCandidate, ...],
    request: dict[str, Any],
    state: dict[str, Any],
    runtime_strategy: dict[str, Any],
) -> dict[str, float]:
    configured = _mapping_from_value(runtime_strategy.get("normalization_refs"))
    state_refs = _mapping_from_value(state.get("normalization_refs"))
    candidate_max = {
        "latency_s": max(
            1.0,
            *(_finite_or(candidate.predicted_e2e_s, 0.0) for candidate in candidates),
            *(_finite_or(candidate.predicted_nominal_latency_s, 0.0) for candidate in candidates),
        ),
        "e2e_s": max(1.0, *(_finite_or(candidate.predicted_e2e_s, 0.0) for candidate in candidates)),
        "ttft_s": max(1.0, *(_finite_or(candidate.predicted_ttft_s, 0.0) for candidate in candidates)),
        "queue_s": max(1.0, *(_finite_or(candidate.predicted_queue_wait_s, 0.0) for candidate in candidates)),
        "link_s": max(1.0, *(_finite_or(candidate.predicted_link_wait_s, 0.0) for candidate in candidates)),
        "energy_risk": max(1.0, *(_finite_or(candidate.predicted_energy_risk, 0.0) for candidate in candidates)),
    }
    latency_default = max(
        1.0,
        *(_finite_or(candidate.predicted_e2e_s, 0.0) for candidate in candidates),
        *(_finite_or(candidate.predicted_nominal_latency_s, 0.0) for candidate in candidates),
    )
    ttft_default = max(
        1.0,
        *(_finite_or(candidate.predicted_ttft_s, 0.0) for candidate in candidates),
    )
    symbolic_values: dict[str, float | None] = {}
    return {
        "latency_s": _normalization_ref(
            configured.get("latency_s"),
            state_refs.get("latency_s"),
            symbolic_values=symbolic_values,
            candidate_max=candidate_max["latency_s"],
            default=latency_default,
        ),
        "e2e_s": _normalization_ref(
            configured.get("e2e_s"),
            state_refs.get("e2e_s"),
            symbolic_values=symbolic_values,
            candidate_max=candidate_max["e2e_s"],
            default=latency_default,
        ),
        "ttft_s": _normalization_ref(
            configured.get("ttft_s"),
            state_refs.get("ttft_s"),
            symbolic_values=symbolic_values,
            candidate_max=candidate_max["ttft_s"],
            default=ttft_default,
        ),
        "queue_s": _normalization_ref(
            configured.get("queue_s"),
            state_refs.get("queue_s"),
            symbolic_values=symbolic_values,
            candidate_max=candidate_max["queue_s"],
            default=latency_default,
        ),
        "link_s": _normalization_ref(
            configured.get("link_s"),
            state_refs.get("link_s"),
            symbolic_values=symbolic_values,
            candidate_max=candidate_max["link_s"],
            default=latency_default,
        ),
        "energy_risk": _normalization_ref(
            configured.get("energy_risk"),
            state_refs.get("energy_risk"),
            symbolic_values=symbolic_values,
            candidate_max=candidate_max["energy_risk"],
            default=1.0,
        ),
        "tokens": _positive_float(configured.get("tokens"), state_refs.get("tokens"), request.get("prompt_tokens"), 1.0),
        "hbm_occupancy": _positive_float(configured.get("hbm_occupancy"), state_refs.get("hbm_occupancy"), 1.0),
    }


def _with_scheduler_gate_results(
    candidate: SchedulerCandidate,
    request: dict[str, Any],
    state: dict[str, Any],
    runtime_strategy: dict[str, Any],
) -> SchedulerCandidate:
    """Make scheduler admission self-contained for direct and runtime callers."""

    results = _mapping_from_value(candidate.hard_gate_results)
    results["physical_eligibility"] = {
        "passed": candidate.feasible,
        "reason": None if candidate.feasible else "candidate_marked_infeasible",
    }
    hard_gate_overrides = _mapping_from_value(runtime_strategy.get("hard_gate_overrides"))
    existing_energy = _mapping_from_value(results.get("energy_soc"))
    soc_min = _first_soc_threshold(
        hard_gate_overrides.get("soc_min"),
        hard_gate_overrides.get("energy_soc_min"),
        candidate.provenance.get("soc_min"),
        state.get("soc_min"),
        state.get("energy_soc_min"),
        existing_energy.get("soc_min"),
    )
    predicted_soc_after = _finite_optional(candidate.predicted_soc_after)
    if soc_min is not None:
        soc_passed = predicted_soc_after is not None and predicted_soc_after >= soc_min
        results["energy_soc"] = {
            "passed": soc_passed,
            "reason": None if soc_passed else "energy_soc",
            "predicted_soc_after": predicted_soc_after,
            "soc_min": soc_min,
            "soc_slack": None if predicted_soc_after is None else predicted_soc_after - soc_min,
        }
    scheduler_owned_reasons = {
        "candidate_marked_infeasible",
    }
    if soc_min is not None:
        scheduler_owned_reasons.add("energy_soc")
    violations = [
        reason for reason in candidate.hard_gate_violations if reason not in scheduler_owned_reasons
    ]
    for gate_name, result in results.items():
        payload = _mapping_from_value(result)
        reason = _optional_str(payload.get("reason"))
        if payload.get("passed") is False:
            reason = reason or f"{gate_name}_failed"
            if reason not in violations:
                violations.append(reason)
    return replace(
        candidate,
        hard_gate_violations=tuple(violations),
        hard_gate_results=to_jsonable(results),
    )


def _finite_optional(value: Any) -> float | None:
    try:
        numeric = _optional_float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric is not None and math_is_finite(numeric) else None


def _first_soc_threshold(*values: Any) -> float | None:
    for value in values:
        numeric = _finite_optional(value)
        if numeric is not None and 0.0 <= numeric <= 1.0:
            return numeric
    return None


def _normalization_ref(
    *values: Any,
    symbolic_values: dict[str, float | None],
    candidate_max: float,
    default: float,
) -> float:
    for value in values:
        if value == "candidate_max":
            return candidate_max
        if isinstance(value, str) and value in symbolic_values:
            symbolic = symbolic_values[value]
            if symbolic is not None and symbolic > 0.0:
                return symbolic
        numeric = _finite_optional(value)
        if numeric is not None and numeric > 0.0:
            return numeric
    return _positive_float(default)


def _score_component(
    raw_metric: Any,
    reference: Any,
    normalized_metric: float,
    weights: dict[str, float],
    weight_key: str,
    default_weight: float,
    weighted_components: dict[str, float],
) -> dict[str, Any]:
    weight = _weight(weights, weight_key, default_weight)
    return {
        "raw_metric": raw_metric,
        "shared_reference": reference,
        "normalized_metric": normalized_metric,
        "weight": weight,
        "weighted_contribution": weighted_components[weight_key],
    }


def _hard_gate_mode(value: Any) -> str:
    mode = str(value or "strict").strip().lower()
    if mode in {"strict", "admission_only", "audit_only"}:
        return mode
    return "strict"


def _hard_gate_blocks_selection(mode: str) -> bool:
    return mode in {"strict", "admission_only"}


def _gate_results(
    candidates: tuple[SchedulerCandidate, ...],
    *,
    hard_gate_mode: str,
    hard_gate_blocking: bool,
) -> dict[str, Any]:
    violations_by_gate: dict[str, int] = {}
    for candidate in candidates:
        for violation in candidate.hard_gate_violations:
            violations_by_gate[violation] = violations_by_gate.get(violation, 0) + 1
    return to_jsonable(
        {
            "hard_gate_mode": hard_gate_mode,
            "hard_gate_blocking": hard_gate_blocking,
            "candidate_count": len(candidates),
            "violation_candidate_count": sum(1 for candidate in candidates if candidate.hard_gate_violations),
            "violations_by_gate": violations_by_gate,
        }
    )


def _rejected_candidate_payloads(
    scored: list[tuple[float, SchedulerCandidate, dict[str, Any]]],
    *,
    selected_candidate_id: str | None,
    hard_gate_blocking: bool,
) -> tuple[dict[str, Any], ...]:
    rejected: list[dict[str, Any]] = []
    for score, candidate, score_terms in scored:
        if selected_candidate_id is not None and candidate.candidate_id == selected_candidate_id:
            continue
        reason = "not_selected"
        if not candidate.feasible:
            reason = "candidate_marked_infeasible"
        elif hard_gate_blocking and candidate.hard_gate_violations:
            reason = "hard_gate_violation"
        rejected.append(
            to_jsonable(
                {
                    "candidate_id": candidate.candidate_id,
                    "action_type": candidate.action_type,
                    "rejection_reason": reason,
                    "score": score,
                    "hard_gate_violations": candidate.hard_gate_violations,
                    "hard_gate_results": candidate.hard_gate_results,
                    "score_terms": score_terms,
                    "provenance": candidate.provenance,
                }
            )
        )
    return tuple(rejected)


def _positive_float(*values: Any) -> float:
    for value in values:
        try:
            numeric = _optional_float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and math_is_finite(numeric) and numeric > 0.0:
            return numeric
    return 1.0


def _finite_or(value: Any, default: float) -> float:
    try:
        numeric = _optional_float(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is None or not math_is_finite(numeric):
        return default
    return numeric


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _weight(weights: dict[str, Any], key: str, default: float) -> float:
    try:
        value = weights.get(key)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _unique_policies(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        policy = _canonical_policy(str(value))
        if policy not in seen:
            seen.add(policy)
            result.append(policy)
    return tuple(result)


def _set_if_not_none(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _set_if_not_empty(payload: dict[str, Any], key: str, value: Any) -> None:
    if value:
        payload[key] = value
