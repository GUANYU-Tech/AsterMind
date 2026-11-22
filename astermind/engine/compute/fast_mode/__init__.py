"""Fast Mode uses grouped Roofline compute and the V1 communication backend."""

__all__ = ("FastModeExecutor",)


def __getattr__(name):
    if name == "FastModeExecutor":
        from .executor import FastModeExecutor

        return FastModeExecutor
    raise AttributeError(name)
