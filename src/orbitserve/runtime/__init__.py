"""Runtime contracts, scenario construction, and evaluation."""

from .contracts import (
    RuntimeAction,
    RuntimeDecisionTrace,
    RuntimePerformanceResult,
    RuntimePredictionEnvelope,
    RuntimeScenario,
    RuntimeStrategySpec,
    ServingDeploymentView,
)
from .evaluation import evaluate_runtime_performance
from .state.energy_accounting import EnergyIntensity, build_planned_energy_demand
from .metrics.energy_alignment import attach_energy_alignment, summarize_runtime_sunlight
from .metrics.result_builder import build_runtime_performance_result
from .scenario.runtime_deployments import build_runtime_deployment_catalog
from .scenario.runtime_energy_envelope import apply_runtime_energy_envelope
from .strategy.runtime_optimization import (
    RuntimeOptimizationObjective,
    optimize_runtime_strategy,
    score_runtime_evaluator_result,
)
from .metrics.runtime_records import RuntimeRequestRecord
from .scenario.scenario import build_runtime_scenario
from .strategy.schedulers import SchedulerSpec, get_scheduler

__all__ = [
    "EnergyIntensity",
    "RuntimeAction",
    "RuntimeDecisionTrace",
    "RuntimeOptimizationObjective",
    "RuntimePerformanceResult",
    "RuntimeRequestRecord",
    "RuntimePredictionEnvelope",
    "RuntimeScenario",
    "RuntimeStrategySpec",
    "SchedulerSpec",
    "ServingDeploymentView",
    "apply_runtime_energy_envelope",
    "attach_energy_alignment",
    "build_planned_energy_demand",
    "build_runtime_deployment_catalog",
    "build_runtime_performance_result",
    "build_runtime_scenario",
    "evaluate_runtime_performance",
    "get_scheduler",
    "optimize_runtime_strategy",
    "score_runtime_evaluator_result",
    "summarize_runtime_sunlight",
]
