"""Dispatch independent terrestrial baseline evaluations."""

from orbitserve.baselines.ground.contracts import GroundScenario
from orbitserve.baselines.ground.evaluation import evaluate_ground_scenario


def evaluate_runtime_performance(scenario, strategy=None):
    if isinstance(scenario, GroundScenario):
        return evaluate_ground_scenario(scenario, strategy)
    raise TypeError("expected GroundScenario")
