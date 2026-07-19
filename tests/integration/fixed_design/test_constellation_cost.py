from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from orbitserve.architecture.hardware import resolve_experiment_hardware
from orbitserve.experiments import load_experiment_spec
from orbitserve.experiments.constellation_cost import estimate_constellation_cost


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/fixtures/manifests/smoke_clustered_constellation.json"


class ConstellationCostTests(unittest.TestCase):
    def _estimate(self, spec):
        hardware = resolve_experiment_hardware(spec.hardware, base_dir=ROOT)
        return estimate_constellation_cost(
            spec.constellation,
            spec.costing,
            cost_limits=hardware.cost_limits,
            cost_config=hardware.cost_config,
            single_satellite_case_path=hardware.single_satellite_case_path,
        )

    def test_constellation_cost_scales_immutable_single_satellite_cost(self) -> None:
        spec = load_experiment_spec(MANIFEST)
        result = self._estimate(spec)
        self.assertTrue(result.evaluated)
        self.assertEqual(result.satellite_count, spec.constellation.satellite_count)
        self.assertAlmostEqual(
            result.total_cost_usd,
            result.single_satellite["C_sat_usd"] * spec.constellation.satellite_count,
        )

    def test_cost_result_is_json_serializable(self) -> None:
        result = self._estimate(load_experiment_spec(MANIFEST))
        json.dumps(result.to_dict())

    def test_orbit_geometry_does_not_rewrite_single_satellite_hardware(self) -> None:
        spec = load_experiment_spec(MANIFEST)
        first = self._estimate(spec)
        changed_design = replace(
            spec.constellation,
            name="same_count_different_orbit",
        )
        hardware = resolve_experiment_hardware(spec.hardware, base_dir=ROOT)
        second = estimate_constellation_cost(
            changed_design,
            spec.costing,
            cost_limits=hardware.cost_limits,
            cost_config=hardware.cost_config,
        )
        self.assertEqual(first.single_satellite, second.single_satellite)


if __name__ == "__main__":
    unittest.main()
