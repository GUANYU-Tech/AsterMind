"""Bounded memoization for pure, immutable native kernel predictions.

No execution timestamps are cached. Keys serialize the complete KernelSpec and
current HWSpec state on every call, including mapping order and nested values.
The serving adapter copies this dependency-free module into local midlayer;
third-party sources themselves stay outside version control.
"""

from collections import OrderedDict, defaultdict
import copyreg
from dataclasses import fields, is_dataclass
from enum import Enum
import io
import pickle
from time import perf_counter
from types import MappingProxyType


def _mapping_proxy(items):
    return MappingProxyType(dict(items))


def _reduce_proxy(value):
    return _mapping_proxy, (tuple(value.items()),)


class _KeyPickler(pickle.Pickler):
    dispatch_table = {**copyreg.dispatch_table, MappingProxyType: _reduce_proxy}


def _key_bytes(value):
    stream = io.BytesIO()
    _KeyPickler(stream, protocol=5).dump(value)
    return stream.getvalue()


def _immutable(value):
    if value is None or type(value) in (str, bytes, bool, int, float) or isinstance(value, Enum):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_immutable(item) for item in value)
    if isinstance(value, MappingProxyType):
        return all(_immutable(k) and _immutable(v) for k, v in value.items())
    if is_dataclass(value) and not isinstance(value, type):
        return value.__dataclass_params__.frozen and all(
            _immutable(getattr(value, field.name)) for field in fields(value))
    return False


class PredictionCache:
    """Runtime-local LRU; unknown inputs and mutable results use the predictor.

    Pickle is used only to produce equality keys from trusted in-process objects;
    no pickle data is loaded. A custom dispatch table handles read-only mappings
    without changing the process-wide pickle registry.
    """

    def __init__(self, max_entries=2048):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 0:
            raise ValueError('prediction cache capacity must be a nonnegative integer')
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hardware_states = OrderedDict()
        self.hits = self.misses = self.bypasses = self.evictions = 0
        self.key_wall_s = self.predictor_wall_s = 0.
        self.by_kind = defaultdict(lambda: [0., 0, 0])

    def __call__(self, predictor, kernel, hardware):
        started = perf_counter()
        kind = str(getattr(getattr(kernel, 'kind', None), 'value', 'unknown'))
        hit = False
        try:
            key = None
            if self.max_entries:
                key_started = perf_counter()
                try:
                    # Only the native HWSpec and dataclass fixtures have a known
                    # complete default state representation. Opaque providers do
                    # not participate in memoization.
                    native_hw = (type(hardware).__module__ == 'midlayer.kernel.hardware_spec'
                                 and type(hardware).__name__ == 'HWSpec')
                    if not native_hw and not is_dataclass(hardware):
                        raise TypeError('opaque hardware provider')
                    state = _key_bytes((type(hardware), vars(hardware)))
                    canonical = self.hardware_states.get(state)
                    if canonical is None:
                        self.hardware_states[state] = state
                        canonical = state
                        if len(self.hardware_states) > 32:
                            self.hardware_states.popitem(last=False)
                    else:
                        self.hardware_states.move_to_end(state)
                    key = predictor, canonical, _key_bytes(kernel)
                except (TypeError, ValueError, AttributeError, pickle.PicklingError, RecursionError):
                    self.bypasses += 1
                finally:
                    self.key_wall_s += perf_counter() - key_started
            if key is not None and key in self.entries:
                result = self.entries[key]
                self.entries.move_to_end(key)
                self.hits += 1
                hit = True
                return result
            self.misses += 1
            prediction_started = perf_counter()
            try:
                result = predictor(kernel, hardware)
            finally:
                self.predictor_wall_s += perf_counter() - prediction_started
            if key is not None and _immutable(result):
                self.entries[key] = result
                if len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
                    self.evictions += 1
            return result
        finally:
            row = self.by_kind[kind]
            row[0] += perf_counter() - started
            row[1] += 1
            row[2] += int(hit)

    def info(self):
        return {'entries': len(self.entries), 'hits': self.hits, 'misses': self.misses,
                'bypasses': self.bypasses, 'evictions': self.evictions,
                'key_wall_s': self.key_wall_s, 'predictor_wall_s': self.predictor_wall_s,
                'by_kind': dict(self.by_kind)}
