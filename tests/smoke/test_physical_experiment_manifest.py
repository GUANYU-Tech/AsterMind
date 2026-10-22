import json
import tempfile
import unittest
from orbitserve.experiments import ExperimentSpec, run_experiment
from tests.system_case import physical_case


class PhysicalExperimentManifestTests(unittest.TestCase):
    def test_manifest_uses_separate_network_and_current_public_modules(self):
        spec = physical_case()
        payload = spec.to_dict()
        self.assertIn("physical_network", payload)
        self.assertIn("communication_hardware", payload)
        self.assertEqual(
            len(spec.physical_network["constellations"]["communication_shells"]), 1
        )
        json.dumps(payload)

    def test_removed_simulation_and_arc_parameters_are_rejected(self):
        for key in ("simulation", "scheduler", "constellation"):
            payload = physical_case().to_dict()
            payload[key] = {}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                ExperimentSpec.from_dict(payload, base_dir=physical_case().base_dir)

    def test_unknown_network_fields_and_execution_worker_setting_fail(self):
        for key in ("workers", "execution"):
            payload = physical_case().to_dict()
            payload["physical_network"][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                ExperimentSpec.from_dict(payload, base_dir=physical_case().base_dir)

    def test_no_paper_promotion_from_synthetic_sgp4_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(physical_case(), output_dir=folder, _workers=1)
            self.assertFalse(result.runtime.provenance["valid_for_paper_main_results"])
            self.assertTrue(
                result.runtime.provenance["trace_model"].startswith("sgp4-")
            )
