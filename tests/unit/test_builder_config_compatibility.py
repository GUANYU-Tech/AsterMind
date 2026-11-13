"""Relocated config classes retain old imports, pickle identities and hints."""
import importlib
import pickle
from typing import get_type_hints
import unittest


class BuilderConfigCompatibilityTests(unittest.TestCase):
    def test_old_and_canonical_imports_are_the_same_classes(self):
        for old, new, names in (
            ('astermind.builder.models', 'astermind.builder.config.resources', ('Cell','Satellite','Device','Link','Scenario')),
            ('astermind.builder.config', 'astermind.builder.config.deployment', ('Parallelism','Placement','ServingCell')),
            ('astermind.builder.config', 'astermind.builder.config.requests', ('Request','Migration')),
            ('astermind.builder.config', 'astermind.builder.config.scenario', ('CellLinkAddition','SimulationConfig')),
            ('astermind.builder.scene', 'astermind.builder.config.deployment', ('SceneDeployment',)),
            ('astermind.engine.setup.network_spec', 'astermind.builder.config.network', ('NetworkNodeSpec','LinkProfile','NetworkSpec')),
            ('astermind.engine.comm.frame_surrogate', 'astermind.builder.config.communication', ('CommPolicy',)),
            ('astermind.engine.comm.policy', 'astermind.builder.config.communication', ('CellCommPolicy',)),
        ):
            for name in names:
                with self.subTest(old=old, name=name):
                    legacy, canonical = importlib.import_module(old), importlib.import_module(new)
                    cls = getattr(legacy, name)
                    self.assertIs(cls, getattr(canonical, name))
                    self.assertEqual(cls.__module__, old)
                    self.assertIs(pickle.loads(pickle.dumps(cls)), cls)
                    self.assertTrue(get_type_hints(cls, include_extras=True))

    def test_flat_config_exports_remain_discoverable(self):
        import astermind.builder.config as config
        for name in ('SimulationConfig','ServingCell','Request','Migration','Parallelism','Placement','CellLinkAddition'):
            self.assertIn(name, dir(config))


if __name__ == '__main__':
    unittest.main()
