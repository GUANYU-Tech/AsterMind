from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orbitserve.architecture.hardware import (
    HardwareSpec,
    SatelliteArchitectureProfile,
    SatelliteEnergyProfile,
    SatelliteHardwareProfile,
    hardware_from_single_satellite_case,
    resolve_satellite_architecture,
)


ROOT = Path(__file__).resolve().parents[3]


class HardwareProfileTests(unittest.TestCase):
    def test_cost_case_uses_explicit_static_quantities(self) -> None:
        profile = hardware_from_single_satellite_case(
            ROOT / "examples/single_satellite/h200_gpu_derived.json"
        )
        self.assertEqual(profile.accelerator_count, 3)
        self.assertEqual(profile.laser_terminal_count, 4)

    def test_fixed_hardware_projects_one_static_source_to_cost_and_performance(self) -> None:
        spec = HardwareSpec.from_dict(
            {
                "mode": "fixed",
                "name": "fixed_satellite",
                "accelerator": "h200_sxm",
                "compute_payload": {
                    "accelerator_count": 2,
                    "purchase_cost_per_unit_usd": 30000,
                },
                "storage": {"profile": "nvme_8tb", "device_count": 2},
                "intra_interconnect": {"switch_count": 1, "switch_bw_gbs": 900},
                "communication": {
                    "ground_link_bandwidth_gbps": 2,
                    "isl_terminal_count": 4,
                    "isl_terminal_bandwidth_gbps": 20,
                },
                "power": {
                    "battery_capacity_wh": 5000,
                    "solar_array_power_w": 6000,
                },
            }
        )
        resolved = resolve_satellite_architecture(spec)
        self.assertIsInstance(resolved, SatelliteArchitectureProfile)
        self.assertEqual(resolved.architecture_id, "fixed_satellite")
        self.assertEqual(resolved.hardware.accelerator_count, 2)
        self.assertEqual(resolved.hardware.storage_capacity_tb, 16)
        self.assertEqual(resolved.hardware.inter_satellite_bandwidth_gbps, 80)
        self.assertIsInstance(resolved.energy, SatelliteEnergyProfile)
        self.assertEqual(resolved.energy.battery_capacity_wh, 5000)
        self.assertNotIn("provenance", resolved.to_dict())
        self.assertNotIn("warnings", resolved.to_dict())
        _limits, cost_config, _path = resolved.cost_inputs()
        self.assertEqual(cost_config.compute.accelerator_count, 2)
        self.assertEqual(cost_config.storage.device_count, 2)

    def test_hardware_resolution_does_not_accept_constellation_inputs(self) -> None:
        spec = HardwareSpec.from_dict(
            {
                "mode": "fixed",
                "compute_payload": {
                    "accelerator_count": 1,
                    "purchase_cost_per_unit_usd": 1,
                },
            }
        )
        with self.assertRaises(TypeError):
            resolve_satellite_architecture(spec, constellation=object())

    def test_source_case_is_an_explicit_top_level_hardware_source(self) -> None:
        spec = HardwareSpec.from_dict(
            {
                "mode": "from_cost_case",
                "source_case_path": "examples/single_satellite/edge_orin_small_sat.json",
            }
        )
        resolved = resolve_satellite_architecture(spec, base_dir=ROOT)
        self.assertEqual(resolved.hardware.accelerator_count, 2)

    def test_missing_hardware_is_explicitly_unevaluated(self) -> None:
        self.assertIsNone(resolve_satellite_architecture(None))

    def test_only_fixed_and_from_cost_case_modes_are_supported(self) -> None:
        spec = HardwareSpec.from_dict(
            {
                "mode": "capability_target",
                "design_limits": {},
            }
        )
        with self.assertRaises(ValueError):
            resolve_satellite_architecture(spec)

    def test_canonical_communication_names_are_serialized(self) -> None:
        spec = HardwareSpec.from_dict(
            {
                "mode": "fixed",
                "compute_payload": {
                    "accelerator_count": 1,
                    "purchase_cost_per_unit_usd": 1,
                },
                "communication": {
                    "ground_link_bandwidth_gbps": 3,
                    "isl_terminal_count": 2,
                    "isl_terminal_bandwidth_gbps": 4,
                },
            }
        )
        payload = spec.communication.to_dict()
        self.assertEqual(payload["ground_link_bandwidth_gbps"], 3)
        self.assertEqual(payload["isl_terminal_bandwidth_gbps"], 4)
        self.assertNotIn("ground_link_capacity_gbps", payload)
        self.assertNotIn("isl_terminal_capacity_gbps", payload)


if __name__ == "__main__":
    unittest.main()
