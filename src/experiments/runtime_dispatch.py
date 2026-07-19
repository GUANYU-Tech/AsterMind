"""Select core orbit runtime or an external baseline evaluator."""

from __future__ import annotations

from typing import Any

from orbitserve.baselines.ground.evaluator import evaluate_ground_runtime_performance
from orbitserve.baselines.ground.global_runtime import is_centralized_ground_policy
from orbitserve.runtime import RuntimeStrategySpec
from orbitserve.runtime import evaluate_runtime_performance as evaluate_orbit_runtime


def evaluate_runtime_performance(scenario: Any, strategy: Any = None) -> Any:
    resolved = RuntimeStrategySpec.from_value(
        strategy or getattr(scenario, "runtime_config", None)
    )
    if resolved.policy in {"ground_regional", "skywalker_like_ground"} or is_centralized_ground_policy(resolved.policy):
        return evaluate_ground_runtime_performance(scenario, resolved)
    return evaluate_orbit_runtime(scenario, resolved)


__all__ = ["evaluate_runtime_performance"]
