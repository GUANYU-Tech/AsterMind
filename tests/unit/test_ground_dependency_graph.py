"""Ground client programs support long chains and multi-parent DAGs."""
import unittest

from orbitserve.baselines.ground.runtime import _validate_program_dependency_dag


class GroundDependencyGraphTests(unittest.TestCase):
    def test_long_dag_cycle_and_unknown_parent(self):
        graph = {f'r{i}': ((f'r{i-1}',) if i else ()) for i in reversed(range(5000))}
        graph['joined'] = ('r4999', 'r2500')
        _validate_program_dependency_dag(graph)
        graph['r0'] = ('joined',)
        with self.assertRaisesRegex(ValueError, 'contains a cycle'):
            _validate_program_dependency_dag(graph)
        with self.assertRaisesRegex(ValueError, 'unknown request'):
            _validate_program_dependency_dag({'request': ('missing',)})
