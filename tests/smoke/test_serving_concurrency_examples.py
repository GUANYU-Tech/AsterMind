import json
import unittest
from pathlib import Path

from orbitserve.architecture import HardwareSpec, resolve_architecture
from orbitserve.serving import ServingSpec, estimate_serving_throughput


ROOT = Path(__file__).resolve().parents[2]


class ServingConcurrencyExampleSmokeTests(unittest.TestCase):
    def test_public_examples_generate_two_dimensional_rate_tables(self):
        hardware = resolve_architecture(HardwareSpec.from_dict({
            "accelerator": "h200_sxm",
            "compute_payload": {"accelerator_count": 4, "purchase_cost_per_unit_usd": 50000},
        })).hardware_profile
        for name in ("fixed_design_geometry", "orbit_geometry_dse"):
            with self.subTest(name=name):
                payload = json.loads(
                    (ROOT / "examples" / "experiments" / f"{name}.json").read_text(encoding="utf-8")
                )
                spec = ServingSpec.from_dict(payload["serving"])
                result = estimate_serving_throughput(spec, hardware=hardware)
                self.assertTrue(result.evaluated)
                self.assertEqual(result.max_concurrency, 8)
                self.assertEqual(
                    {row["concurrency"] for row in result.prefill_rate_table},
                    set(range(1, 9)),
                )


if __name__ == "__main__":
    unittest.main()
