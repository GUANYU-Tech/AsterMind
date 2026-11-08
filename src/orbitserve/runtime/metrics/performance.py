"""Opt-in nested wall timers with exclusive accounting for runtime diagnosis."""

from collections import defaultdict
from functools import wraps
from time import perf_counter


class WallProfiler:
    """One profiler per single-threaded event engine.

    Exclusive time subtracts instrumented children. Uninstrumented work stays
    with its enclosing scope, so the event-loop root provides an explicit
    remainder instead of silently attributing it to a scheduler or transport.
    """

    def __init__(self, root_label=None):
        self.root_label = root_label
        self._stack = []
        self._rows = defaultdict(lambda: [0., 0., 0])

    def wrap(self, label, function):
        @wraps(function)
        def measured(*args, **kwargs):
            if self.root_label is not None and not self._stack and label != self.root_label:
                return function(*args, **kwargs)
            frame = [perf_counter(), 0.]
            self._stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = perf_counter() - frame[0]
                self._stack.pop()
                row = self._rows[label]
                row[0] += elapsed
                row[1] += elapsed - frame[1]
                row[2] += 1
                if self._stack:
                    self._stack[-1][1] += elapsed
        return measured

    def report(self):
        if self._stack:
            raise RuntimeError('cannot export an active performance scope')
        return {label: {'inclusive_wall_s': row[0], 'exclusive_wall_s': row[1],
                        'calls': row[2]} for label, row in self._rows.items()}
