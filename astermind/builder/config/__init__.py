"""Public simulation configuration, organized by configuration domain."""
from importlib import import_module

_EXPORTS = {'Parallelism': 'deployment', 'Placement': 'deployment', 'ServingCell': 'deployment', 'Positive': 'deployment', 'Request': 'requests', 'Migration': 'requests', 'CellLinkAddition': 'scenario', 'SimulationConfig': 'scenario', 'CommPolicy': 'communication'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(__all__))
