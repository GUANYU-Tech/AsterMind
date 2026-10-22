"""Select the satellite Runtime or the independent terrestrial evaluator."""

from dataclasses import replace

from orbitserve.runtime import RuntimeSimulationSpec, run_simulation
from orbitserve.baselines.ground.contracts import GroundScenario
from orbitserve.baselines.ground.evaluation import evaluate_ground_scenario


def evaluate_runtime_performance(scenario, strategy=None):
    if isinstance(scenario, RuntimeSimulationSpec):
        return run_simulation(replace(scenario, strategy=strategy or scenario.strategy))
    if isinstance(scenario, GroundScenario):
        return evaluate_ground_scenario(scenario, strategy)
    raise TypeError("expected RuntimeSimulationSpec or GroundScenario")
