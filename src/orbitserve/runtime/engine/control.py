"""Opt-in in-process control hook, isolated per experiment thread/context."""

from contextlib import contextmanager
from contextvars import ContextVar

_controller = ContextVar('orbitserve_runtime_controller', default=None)
_compute_runtime = ContextVar('orbitserve_compute_runtime', default=None)


def current_controller():
    return _controller.get()


def current_compute_runtime():
    return _compute_runtime.get()


@contextmanager
def compute_runtime_scope(runtime):
    token = _compute_runtime.set(runtime)
    try:
        yield
    finally:
        _compute_runtime.reset(token)


@contextmanager
def runtime_control(controller):
    token = _controller.set(controller)
    try:
        yield
    finally:
        _controller.reset(token)
