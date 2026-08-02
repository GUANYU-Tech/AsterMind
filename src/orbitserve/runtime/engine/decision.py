"""Deterministic request-level compute-satellite selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable
from ..contracts import RuntimeStrategySpec


@dataclass(frozen=True)
class SchedulerCandidate:
    candidate_id: str
    compute_satellite_id: str
    feasible: bool
    route_plan: dict[str, Any]
    latency_score: float
    kv_reuse_score: float
    energy_score: float
    compute_load_score: float
    arc_distance_rad: float
    compute_queue_wait_s: float
    hard_gate_violations: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "SchedulerCandidate":
        if isinstance(value, cls):
            return value
        payload = dict(value)
        return cls(
            candidate_id=str(payload["candidate_id"]),
            compute_satellite_id=str(payload["compute_satellite_id"]),
            feasible=bool(payload.get("feasible", True)),
            route_plan=dict(payload.get("route_plan") or {}),
            latency_score=_score(payload.get("latency_score")),
            kv_reuse_score=_score(payload.get("kv_reuse_score")),
            energy_score=_score(payload.get("energy_score")),
            compute_load_score=_score(payload.get("compute_load_score")),
            arc_distance_rad=max(0.0, float(payload.get("arc_distance_rad", 0.0))),
            compute_queue_wait_s=max(0.0, float(payload.get("compute_queue_wait_s", 0.0))),
            hard_gate_violations=tuple(str(item) for item in payload.get("hard_gate_violations", ())),
            provenance=dict(payload.get("provenance") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class SchedulerDecision:
    evaluated: bool
    selected_candidate_id: str | None
    score: float | None
    score_terms: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0
    feasible_candidate_count: int = 0
    rejected_candidates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class CandidateSelector:
    spec: RuntimeStrategySpec

    def select_action(self, state: Any, request: Any = None) -> SchedulerDecision:
        del request
        payload = state if isinstance(state, dict) else {}
        candidates = tuple(
            SchedulerCandidate.from_value(item)
            for item in payload.get("candidates", ())
        )
        feasible = tuple(
            candidate
            for candidate in candidates
            if candidate.feasible and not candidate.hard_gate_violations
        )
        if not feasible:
            return SchedulerDecision(
                evaluated=False,
                selected_candidate_id=None,
                score=None,
                candidate_count=len(candidates),
                feasible_candidate_count=0,
                rejected_candidates=tuple(_rejected(candidate) for candidate in candidates),
            )
        scored = tuple((self._score(candidate), candidate) for candidate in feasible)
        score, selected = min(
            scored,
            key=lambda item: (
                -item[0],
                item[1].arc_distance_rad,
                item[1].compute_queue_wait_s,
                item[1].compute_satellite_id,
                item[1].candidate_id,
            ),
        )
        return SchedulerDecision(
            evaluated=True,
            selected_candidate_id=selected.candidate_id,
            score=score,
            score_terms={
                "latency_score": selected.latency_score,
                "kv_reuse_score": selected.kv_reuse_score,
                "energy_score": selected.energy_score,
                "compute_load_score": selected.compute_load_score,
                "weights": self.spec.weights,
                "weighted_components": {
                    "latency": self.spec.latency_weight * selected.latency_score,
                    "kv_reuse": self.spec.kv_reuse_weight * selected.kv_reuse_score,
                    "energy": self.spec.energy_weight * selected.energy_score,
                    "compute_load": self.spec.compute_load_weight * selected.compute_load_score,
                },
                "score_model": "normalized_compute_satellite_score_v1",
            },
            candidate_count=len(candidates),
            feasible_candidate_count=len(feasible),
            rejected_candidates=tuple(
                _rejected(candidate)
                for candidate in candidates
                if candidate not in feasible
            ),
        )

    def _score(self, candidate: SchedulerCandidate) -> float:
        return (
            self.spec.latency_weight * candidate.latency_score
            + self.spec.kv_reuse_weight * candidate.kv_reuse_score
            + self.spec.energy_weight * candidate.energy_score
            + self.spec.compute_load_weight * candidate.compute_load_score
        )


def build_candidate_selector(spec: RuntimeStrategySpec | dict[str, Any] | None = None) -> CandidateSelector:
    return CandidateSelector(RuntimeStrategySpec.from_value(spec))


def _score(value: Any) -> float:
    parsed = float(value or 0.0)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("candidate scores must be in [0, 1]")
    return parsed


def _rejected(candidate: SchedulerCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "compute_satellite_id": candidate.compute_satellite_id,
        "reasons": candidate.hard_gate_violations or (("candidate_infeasible",) if not candidate.feasible else ()),
    }


__all__ = ["CandidateSelector", "SchedulerCandidate", "SchedulerDecision", "build_candidate_selector"]
