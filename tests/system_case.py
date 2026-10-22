from pathlib import Path
from orbitserve.experiments import load_experiment_spec
ROOT=Path(__file__).resolve().parents[1]
def physical_case():
    return load_experiment_spec(ROOT/"tests/fixtures/manifests/coupled_physical.json")
