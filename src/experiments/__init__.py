"""Top-level experiment framework for OrbitServe."""

from typing import Any

__all__ = [
    "ExperimentResult",
    "ExperimentSpec",
    "load_experiment_spec",
    "run_experiment",
]


def __getattr__(name: str) -> Any:
    if name == "load_experiment_spec":
        from .loader import load_experiment_spec

        return load_experiment_spec
    if name in {"ExperimentResult", "ExperimentSpec"}:
        from .models import ExperimentResult, ExperimentSpec

        return {"ExperimentResult": ExperimentResult, "ExperimentSpec": ExperimentSpec}[name]
    if name == "run_experiment":
        from .runner import run_experiment

        return run_experiment
    raise AttributeError(name)
