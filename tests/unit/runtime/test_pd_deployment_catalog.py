import unittest
from types import SimpleNamespace

from orbitserve.runtime.scenario.runtime_deployments import build_runtime_deployment_catalog


def _scenario(*, satellite_ids=("sat-01", "sat-03")):
    return SimpleNamespace(
        constellation={
            "satellite_catalog": tuple(
                {"satellite_id": satellite_id} for satellite_id in satellite_ids
            )
        },
        serving_deployments=(
            {
                "deployment_id": "deployment_0",
                "throughput_estimate": {
                    "evaluated": True,
                    "feasible": True,
                    "prefill_tokens_per_s": 100.0,
                    "decode_tokens_per_s": 50.0,
                    "request_throughput_per_s": 2.0,
                    "kv_cache_capacity_tokens": 10_000,
                    "kv_cache_bytes_per_token": 16.0,
                },
            },
        ),
    )


class SingleSatelliteServingCatalogTests(unittest.TestCase):
    def test_catalog_exposes_full_serving_members(self):
        catalog = build_runtime_deployment_catalog(_scenario())

        deployment = catalog.default_deployment
        self.assertIsNotNone(deployment)
        self.assertEqual(deployment.member_satellites, ("sat-01", "sat-03"))
        self.assertEqual(deployment.entry_satellite_ids, ("sat-01", "sat-03"))
        self.assertEqual(deployment.prefill_tokens_per_s, 100.0)
        self.assertEqual(deployment.decode_tokens_per_s, 50.0)
        self.assertEqual(catalog.placement_model, "single_satellite_full_serving")
        self.assertEqual(catalog.selection_policy, "runtime_scheduler_selects_serving_satellite")

    def test_catalog_does_not_synthesize_missing_orbit_satellites(self):
        catalog = build_runtime_deployment_catalog(_scenario(satellite_ids=()))

        self.assertIsNone(catalog.default_deployment)
        self.assertIn("Orbit satellite_catalog", " ".join(catalog.warnings))

    def test_catalog_contract_has_no_phase_separation(self):
        payload = build_runtime_deployment_catalog(_scenario()).to_dict()
        deployment = payload["default_deployment"]

        for key in (
            "role",
            "pd_mode",
            "paired_deployment_ids",
            "paired_deployment_scopes",
            "paired_deployment_rules",
        ):
            self.assertNotIn(key, deployment)
        self.assertNotIn("prefill_deployments", payload)
        self.assertNotIn("decode_deployments", payload)


if __name__ == "__main__":
    unittest.main()
