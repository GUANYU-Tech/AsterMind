"""Reject unsupported numeric inputs before evaluating a scientific model."""
import unittest

from orbitserve.serving import L2FormationSpec, ServingParallelism, ServingSpec


class ServingInputValidationTests(unittest.TestCase):
    def test_parallel_degrees_and_collective_counts_require_integers(self):
        for key in ('tensor_parallel_degree', 'pipeline_parallel_degree',
                    'expert_parallel_degree', 'tensor_parallel_collectives_per_layer'):
            for value in (1.5, True, float('nan'), '2'):
                with self.subTest(field=key, value=value), self.assertRaisesRegex(ValueError, key):
                    ServingParallelism.from_dict({key: value})

    def test_communication_coefficients_must_be_finite(self):
        for key in ('expert_parallel_imbalance_factor', 'expert_parallel_startup_us'):
            for value in (float('nan'), float('inf'), True, -1.):
                with self.subTest(field=key, value=value), self.assertRaisesRegex(ValueError, key):
                    ServingParallelism(**{key: value})

    def test_formation_and_length_grids_do_not_truncate_fractions(self):
        for value in (1.5, True, '1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'member_count'):
                L2FormationSpec.from_dict({'member_count': value})
        for key in ('prompt_length_grid', 'context_length_grid'):
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, key):
                ServingSpec.from_dict({key: [1, 2.5, 4]})
