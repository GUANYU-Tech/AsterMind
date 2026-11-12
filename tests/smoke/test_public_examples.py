"""Public native examples retain their typed configuration and topology."""
from pathlib import Path
import unittest
from simulator.config import SimulationConfig
ROOT = Path(__file__).resolve().parents[2]
class PublicExampleTests(unittest.TestCase):
    def test_native_examples_validate(self):
        for path in (ROOT / "simulator/examples").glob("*.json"):
            if path.name == "dynamic_physics.json":
                continue
            with self.subTest(path=path.name):
                config = SimulationConfig.load(path)
                self.assertTrue(config.topology().devices)
                self.assertTrue(config.cells)
