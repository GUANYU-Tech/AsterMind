"""Opt-in in-process control hook, isolated per experiment thread/context."""

from contextlib import contextmanager
from contextvars import ContextVar

_controller = ContextVar('orbitserve_runtime_controller', default=None)


def current_controller():
    return _controller.get()


@contextmanager
def runtime_control(controller):
    token = _controller.set(controller)
    try:
        yield
    finally:
        _controller.reset(token)
