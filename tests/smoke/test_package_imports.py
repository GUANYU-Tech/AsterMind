"""Every configured core package and orchestration module must import."""

import importlib
import pkgutil
import unittest
import astermind


class PackageImportTests(unittest.TestCase):
    def test_current_modules_have_no_dangling_cross_module_imports(self):
        from astermind.common.bootstrap import activate
        activate()
        modules = [entry.name for entry in pkgutil.walk_packages(astermind.__path__, 'astermind.')]
        for name in modules:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))
