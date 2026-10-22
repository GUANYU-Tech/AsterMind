"""Every configured core package and orchestration module must import."""

import importlib
import unittest


class PackageImportTests(unittest.TestCase):
    def test_current_modules_have_no_dangling_cross_module_imports(self):
        modules = (
            "orbitserve.architecture",
            "orbitserve.orbit",
            "orbitserve.workload",
            "orbitserve.serving",
            "orbitserve.runtime",
            "orbitserve.experiments.models",
            "orbitserve.experiments.runner",
            "orbitserve.experiments.cli",
            "orbitserve.dse.search",
            "orbitserve.dse.runtime_matrix",
            "orbitserve.dse.static_slo",
            "orbitserve.dse.satellite_budget",
            "orbitserve.baselines.ground.runtime",
            "orbitserve.baselines.ground.global_runtime",
        )
        for name in modules:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))
