"""Exact input keys must not hide changed KV/MoE/hardware state or failures."""

from dataclasses import dataclass, field
from types import MappingProxyType
import unittest

from orbitserve.runtime.prediction_cache import PredictionCache


@dataclass(frozen=True)
class Spec:
    attributes: object = field(default_factory=dict)


@dataclass
class Hardware:
    bandwidth: float = 80.
    nested: object = field(default_factory=lambda: {'clock': 1.})


@dataclass(frozen=True)
class Prediction:
    duration: float
    breakdown: object


def predict(spec, hardware):
    return Prediction(hardware.bandwidth + sum(spec.attributes['experts']) + spec.attributes['kv'],
                      MappingProxyType({'clock': hardware.nested['clock']}))


class PredictionCacheTests(unittest.TestCase):
    def test_equivalent_inputs_reuse_full_immutable_result(self):
        cache = PredictionCache()
        spec = Spec(MappingProxyType({'experts': (1, 3), 'kv': 32}))
        first = cache(predict, spec, Hardware())
        second = cache(predict, Spec(MappingProxyType(dict(spec.attributes))), Hardware())
        self.assertIs(first, second)
        with self.assertRaises(TypeError):
            second.breakdown['clock'] = 3
        self.assertEqual(cache.info()['hits'], 1)

    def test_nested_hardware_kv_and_expert_changes_are_not_hidden(self):
        cache = PredictionCache()
        hardware = Hardware()
        attrs = {'experts': [1, 3], 'kv': 32}
        spec = Spec(attrs)
        values = [cache(predict, spec, hardware)]
        hardware.bandwidth = 40.
        values.append(cache(predict, spec, hardware))
        hardware.nested['clock'] = 2.
        values.append(cache(predict, spec, hardware))
        attrs['kv'] = 64
        values.append(cache(predict, spec, hardware))
        attrs['experts'][0] = 9
        values.append(cache(predict, spec, hardware))
        self.assertEqual(cache.info()['hits'], 0)
        self.assertEqual(len(set((v.duration, v.breakdown['clock']) for v in values)), 5)

    def test_mutable_results_are_not_shared_and_exceptions_are_not_cached(self):
        cache = PredictionCache()
        spec = Spec({'experts': (1,), 'kv': 2})
        def fn(*args):
            return {'value': []}
        first = cache(fn, spec, Hardware())
        first['value'].append(3)
        self.assertEqual(cache(fn, spec, Hardware()), {'value': []})
        def failed(*args):
            raise ValueError('unsupported')
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                cache(failed, spec, Hardware())
        self.assertEqual(cache.info()['entries'], 0)

    def test_capacity_bounds_retention(self):
        cache = PredictionCache(1)
        for kv in (1, 2, 1):
            cache(predict, Spec({'experts': (1,), 'kv': kv}), Hardware())
        self.assertEqual(cache.info()['entries'], 1)
        self.assertEqual(cache.info()['evictions'], 2)


if __name__ == '__main__':
    unittest.main()
