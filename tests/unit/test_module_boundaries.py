"""The Builder/Engine layout has explicit boundaries and no retired packages."""
import ast
import unittest
from pathlib import Path

import astermind.builder.orbit as orbit
import astermind.engine as engine
import astermind.workload as workload

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / 'astermind'


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.ImportFrom):
            yield node.module or ''
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


class CoreModuleBoundaryTests(unittest.TestCase):
    def test_public_contracts_are_exported(self):
        for module, names in {
            orbit: ('OrbitTraceManifest', 'OrbitTraceBundle', 'generate_orbit_trace', 'load_orbit_trace'),
            workload: ('WorkloadBundleRef', 'WorkloadGenerationResult', 'materialize_workload_bundle'),
            engine: ('NetworkSpec', 'NetworkState', 'GlobalTimeline'),
        }.items():
            for name in names:
                with self.subTest(module=module.__name__, name=name):
                    self.assertTrue(hasattr(module, name))

    def test_core_domain_imports_follow_one_way_allowlist(self):
        # Builder validation/loading intentionally instantiate Engine objects.
        # Pure contracts, workload, orbit and topology must stay independent.
        allowed = {
            'common': ('astermind.common',),
            'workload': ('astermind.common', 'astermind.workload'),
            'builder/orbit': ('astermind.common', 'astermind.builder.orbit'),
            'builder/topology': ('astermind.common', 'astermind.builder.models', 'astermind.builder.topology'),
        }
        for folder, prefixes in allowed.items():
            for path in (CORE / folder).rglob('*.py'):
                for module in imports(path):
                    if module.startswith('astermind.'):
                        self.assertTrue(any(module == p or module.startswith(p + '.') for p in prefixes),
                                        f'{path.relative_to(ROOT)} imports {module}')

    def test_cross_domain_imports_use_public_package_interfaces(self):
        # Orbit and workload keep their existing public domain interfaces.
        for path in CORE.rglob('*.py'):
            for module in imports(path):
                for prefix, folder in (('astermind.builder.orbit', CORE / 'builder/orbit'),
                                       ('astermind.workload', CORE / 'workload')):
                    if not path.is_relative_to(folder):
                        self.assertFalse(module.startswith(prefix + '.'), f'{path}: {module}')

    def test_all_source_consumers_use_core_public_package_interfaces(self):
        retired = ('orbitserve', 'simulator', 'network_sim', 'integration')
        for folder in (CORE, ROOT / 'viz', ROOT / 'workloads'):
            for path in folder.rglob('*.py'):
                for module in imports(path):
                    self.assertNotIn(module.split('.')[0], retired, f'{path}: {module}')
        for folder in ('src', 'simulator', 'network_sim', 'integration', 'astermind/runtime', 'astermind/engine/runtime'):
            self.assertFalse(any((ROOT / folder).rglob('*.py')), folder)

    def test_removed_cross_domain_fields_are_absent(self):
        for removed in ('ConstellationDesign', 'RuntimeTraceBundle', 'SimulationConfig', 'run_simulation'):
            self.assertFalse(hasattr(orbit, removed))

    def test_workload_contract_is_unversioned(self):
        self.assertEqual(workload.WORKLOAD_BUNDLE_SCHEMA, 'orbitserve.workload')
        self.assertNotIn('v2', workload.WORKLOAD_BUNDLE_SCHEMA.lower())


if __name__ == '__main__':
    unittest.main()
