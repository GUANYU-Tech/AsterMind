"""Every configured core package and orchestration module must import."""

import importlib
import pkgutil
import unittest
import orbitserve


class PackageImportTests(unittest.TestCase):
    def test_current_modules_have_no_dangling_cross_module_imports(self):
        modules = [entry.name for entry in pkgutil.walk_packages(orbitserve.__path__, 'orbitserve.')]
        for name in modules:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))
