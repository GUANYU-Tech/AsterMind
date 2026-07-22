import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from orbitserve.baselines.ground.costing import estimate_terrestrial_ground_cost
from orbitserve.dse.search import DSERuntimeMatrixSpec
from orbitserve.dse.runtime_matrix import DSEDesignContext, evaluate_runtime_matrix
from orbitserve.runtime import RuntimeScenario, RuntimeStrategySpec
from orbitserve.experiments.runtime_dispatch import evaluate_runtime_performance
from orbitserve.runtime.runtime_records import RuntimeRequestRecord


def _record(
    request_id,
    *,
    origin="us",
    serving="us",
    cross_region=False,
    accepted=True,
    completed=True,
    served=True,
    prompt_tokens=100,
    output_tokens=20,
    prefill_service_s=0.1,
    decode_service_s=0.2,
):
    selected_path = ()
    if cross_region:
        selected_path = (
            {
                "relation": "client_to_regional_lb",
                "source": f"client:{origin}",
                "target": f"lb:{origin}",
                "one_way_delay_s": 0.01,
            },
            {
                "relation": "cross_region_lb_forward",
                "source": f"lb:{origin}",
                "target": f"lb:{serving}",
                "one_way_delay_s": 0.10,
            },
            {
                "relation": "replica_to_client_stream",
                "source": f"ground:{serving}:replica-0",
                "target": f"client:{origin}",
                "one_way_delay_s": 0.10,
            },
        )
    return RuntimeRequestRecord(
        request_id=request_id,
        arrival_time_s=0.0,
        deployment_id=f"ground:{serving}:replica-0",
        action_type="ground_cross_region_forward" if cross_region else "ground_local_compute",
        accepted=accepted,
        completed=completed,
        served=served,
        rejected=not accepted,
        timeout=accepted and completed and not served,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ground_origin_region_id=origin,
        ground_serving_region_id=serving,
        ground_replica_id=f"ground:{serving}:replica-0",
        ground_cross_region=cross_region,
        selected_path=selected_path,
        prefill_service_s=prefill_service_s if completed else None,
        decode_service_s=decode_service_s if completed else None,
    )


def _ground_config(*, include_egress=True):
    costing = {
        "service_period_s": 3600,
        "region_replica_hourly_price_usd": {"us": 0.5, "eu": 1.0},
    }
    if include_egress:
        costing.update(
            {
                "region_egress_cost_per_gb_usd": {"us": 0.1, "eu": 0.2},
                "prompt_bytes_per_token": 4,
                "output_bytes_per_token": 2,
            }
        )
    return {
        "regions": [
            {"region_id": "us", "replica_count": 2},
            {"region_id": "eu", "replica_count": 1},
        ],
        "terrestrial_costing": costing,
    }


class TerrestrialGroundCostTests(unittest.TestCase):
    def test_prices_all_replicas_for_service_period_and_optionally_cross_region_egress(self):
        result = estimate_terrestrial_ground_cost(
            records=(
                _record("local", prompt_tokens=100, output_tokens=50),
                _record(
                    "cross-completed",
                    origin="us",
                    serving="eu",
                    cross_region=True,
                    prompt_tokens=200,
                    output_tokens=100,
                ),
                _record(
                    "cross-rejected-after-forward",
                    origin="eu",
                    serving="us",
                    cross_region=True,
                    accepted=False,
                    completed=False,
                    served=False,
                    prompt_tokens=10,
                    output_tokens=0,
                ),
            ),
            ground_config=_ground_config(),
        )

        # Two US replicas and one EU replica are provisioned for one hour.
        self.assertTrue(result.evaluated)
        self.assertAlmostEqual(result.compute_cost_usd, 2.0)
        # US prompt 200*4, EU response 100*2, EU forwarded-and-rejected
        # prompt 10*4 bytes; GB uses decimal 1e9 bytes.
        expected_egress = (800 / 1e9 * 0.1) + (200 / 1e9 * 0.2) + (40 / 1e9 * 0.2)
        self.assertAlmostEqual(result.egress_cost_usd, expected_egress)
        self.assertAlmostEqual(result.total_cost_usd, 2.0 + expected_egress)
        self.assertAlmostEqual(result.annualized_cost_usd_per_year, result.total_cost_usd * 8760)
        self.assertEqual(result.provisioned_instance_hours, 3.0)
        self.assertEqual(result.completed_request_count, 2)
        self.assertEqual(result.served_request_count, 2)
        self.assertEqual(result.completed_token_count, 450)
        self.assertAlmostEqual(result.cost_per_completed_request_usd, result.total_cost_usd / 2)
        self.assertAlmostEqual(result.cost_per_completed_token_usd, result.total_cost_usd / 450)
        self.assertFalse(result.provenance["leo_cost_reused"])

    def test_egress_is_optional_but_its_exclusion_is_explicit(self):
        result = estimate_terrestrial_ground_cost(
            records=(_record("cross", origin="us", serving="eu", cross_region=True),),
            ground_config=_ground_config(include_egress=False),
        )

        self.assertTrue(result.evaluated)
        self.assertEqual(result.compute_cost_usd, 2.0)
        self.assertEqual(result.egress_cost_usd, 0.0)
        self.assertIsNone(result.egress_total_gb)
        self.assertTrue(any("egress pricing is not configured" in warning for warning in result.warnings))

    def test_missing_any_configured_region_price_fails_closed(self):
        config = _ground_config()
        config["terrestrial_costing"]["region_replica_hourly_price_usd"] = {"us": 0.5}

        result = estimate_terrestrial_ground_cost(
            records=(_record("local"),),
            ground_config=config,
        )

        self.assertFalse(result.evaluated)
        self.assertIsNone(result.total_cost_usd)
        self.assertIsNone(result.annualized_cost_usd_per_year)
        self.assertIsNone(result.cost_per_served_mtoken_usd)
        self.assertTrue(any("missing configured region" in warning for warning in result.warnings))
        self.assertFalse(result.provenance["leo_cost_reused"])

    def test_missing_or_unconfigured_ground_region_fails_closed(self):
        result = estimate_terrestrial_ground_cost(
            records=(_record("unknown-origin", origin="unknown", serving="us"),),
            ground_config=_ground_config(include_egress=False),
        )

        self.assertFalse(result.evaluated)
        self.assertIsNone(result.total_cost_usd)
        self.assertTrue(any("unconfigured ground_origin_region_id" in warning for warning in result.warnings))

    def test_central_ingress_to_remote_replica_egress_uses_selected_path_not_origin_serving_shortcut(self):
        record = RuntimeRequestRecord(
            request_id="central-us-to-eu",
            arrival_time_s=0.0,
            deployment_id="ground:eu:replica-0",
            action_type="ground_central_rr",
            accepted=True,
            completed=True,
            served=True,
            rejected=False,
            timeout=False,
            prompt_tokens=200,
            output_tokens=100,
            ground_origin_region_id="eu",
            ground_serving_region_id="eu",
            ground_replica_id="ground:eu:replica-0",
            # Origin and serving region match, but central US forwards the
            # prompt to EU.  The selected path, rather than this flag, must
            # drive egress accounting.
            ground_cross_region=False,
            selected_path=(
                {
                    "relation": "client_to_baseline_ingress",
                    "source": "client:eu",
                    "target": "lb:us",
                    "one_way_delay_s": 0.10,
                },
                {
                    "relation": "baseline_ingress_to_replica",
                    "source": "lb:us",
                    "target": "ground:eu:replica-0",
                    "one_way_delay_s": 0.10,
                },
                {
                    "relation": "replica_to_client_stream",
                    "source": "ground:eu:replica-0",
                    "target": "client:eu",
                    "one_way_delay_s": 0.01,
                },
            ),
        )
        result = estimate_terrestrial_ground_cost(
            records=(record,), ground_config=_ground_config()
        )

        self.assertTrue(result.evaluated)
        # 200 prompt tokens * 4 bytes leave the central US ingress; no EU
        # response egress is charged because the client is in EU.
        self.assertAlmostEqual(result.egress_cost_usd, 800 / 1e9 * 0.1)
        us_row = next(row for row in result.region_costs if row["region_id"] == "us")
        self.assertEqual(us_row["prompt_egress_bytes"], 800)

    def test_central_same_origin_and_serving_without_path_fails_closed(self):
        """A dropped central path must not silently erase US->EU prompt egress."""

        record = RuntimeRequestRecord(
            request_id="corrupt-central-path",
            arrival_time_s=0.0,
            deployment_id="ground:eu:replica-0",
            action_type="ground_central_rr",
            accepted=True,
            completed=True,
            served=True,
            rejected=False,
            timeout=False,
            prompt_tokens=100,
            output_tokens=20,
            ground_origin_region_id="eu",
            ground_serving_region_id="eu",
            ground_replica_id="ground:eu:replica-0",
            ground_cross_region=False,
            selected_path=(),
            provenance={"baseline_policy": "ground_central_rr"},
        )

        result = estimate_terrestrial_ground_cost(
            records=(record,), ground_config=_ground_config()
        )

        self.assertFalse(result.evaluated)
        self.assertTrue(
            any("requires a non-empty selected_path" in warning for warning in result.warnings)
        )

    def test_central_nonempty_but_malformed_path_fails_closed(self):
        record = RuntimeRequestRecord(
            request_id="corrupt-central-nonempty-path",
            arrival_time_s=0.0,
            deployment_id="ground:eu:replica-0",
            action_type="ground_central_rr",
            accepted=True,
            completed=True,
            served=True,
            rejected=False,
            timeout=False,
            prompt_tokens=100,
            output_tokens=20,
            ground_origin_region_id="eu",
            ground_serving_region_id="eu",
            ground_replica_id="ground:eu:replica-0",
            ground_cross_region=False,
            selected_path=(
                {
                    "relation": "not_a_network_leg",
                    "source": "invalid",
                    "target": "invalid",
                    "one_way_delay_s": 0.0,
                },
            ),
            provenance={"baseline_policy": "ground_central_rr"},
        )

        result = estimate_terrestrial_ground_cost(
            records=(record,), ground_config=_ground_config()
        )

        self.assertFalse(result.evaluated)
        self.assertTrue(
            any("requires exactly one client_to_baseline_ingress" in warning for warning in result.warnings)
        )

    def test_cross_region_record_without_auditable_path_fails_closed_when_egress_is_priced(self):
        record = replace(
            _record("missing-path", origin="us", serving="eu", cross_region=True),
            selected_path=(),
        )
        result = estimate_terrestrial_ground_cost(
            records=(record,),
            ground_config=_ground_config(),
        )

        self.assertFalse(result.evaluated)
        self.assertTrue(
            any("must have exactly one priced prompt" in warning for warning in result.warnings)
        )

    def test_runtime_evaluator_exposes_explicit_ground_cost_without_leo_cost(self):
        ground_config = _ground_config(include_egress=False)
        ground_config["replica_token_service"] = {
            "prefill_tokens_per_s": 1000.0,
            "decode_tokens_per_s": 1000.0,
            "continuous_batch_capacity_tokens": 4096,
            "kv_cache_capacity_tokens": 4096,
        }
        strategy = RuntimeStrategySpec.from_dict(
            {"policy": "ground_regional", "ground_config": ground_config}
        )
        scenario = RuntimeScenario(
            scenario_id="ground-cost-contract",
            requests=(
                {
                    "request_id": "request-0",
                    "arrival_time_s": 0.0,
                    "user_region_id": "us",
                    "prompt_tokens": 100,
                    "output_tokens": 20,
                    "ttft_deadline_s": 10.0,
                    "e2e_deadline_s": 20.0,
                },
            ),
            runtime_config=strategy,
            # Deliberately put a huge LEO number here; it must be ignored.
            cost_power={"annualized_cost_usd_per_year": 9_999_999.0},
        )

        result = evaluate_runtime_performance(scenario)

        self.assertEqual(result.module_status["cost_model"], "terrestrial_cost_evaluated")
        self.assertEqual(result.runtime_metrics["terrestrial_total_cost_usd"], 2.0)
        self.assertEqual(result.runtime_metrics["terrestrial_annualized_cost_usd_per_year"], 17_520.0)
        self.assertEqual(result.runtime_metrics["cost_per_served_request_usd"], 2.0)
        contract = result.details["terrestrial_costing"]
        self.assertTrue(contract["evaluated"])
        self.assertFalse(contract["provenance"]["leo_cost_reused"])
        self.assertNotEqual(
            result.runtime_metrics["terrestrial_annualized_cost_usd_per_year"],
            scenario.cost_power["annualized_cost_usd_per_year"],
        )

    def test_dse_ground_row_preserves_only_explicit_terrestrial_cost_contract(self):
        payload = {
            "evaluated": True,
            "runtime_metrics": {
                "served_ratio": 1.0,
                "useful_request_throughput_per_s": 1.0,
                "useful_token_throughput_per_s": 100.0,
                "cost_per_served_request_usd": 2.0,
                "cost_per_served_mtoken_usd": 20_000.0,
            },
            "slo": {"served_ratio": 1.0, "timeout_ratio": 0.0},
            "performance": {"ttft_p95_s": 1.0, "e2e_p95_s": 2.0},
            "details": {
                "terrestrial_costing": {
                    "evaluated": True,
                    "total_cost_usd": 2.0,
                    "annualized_cost_usd_per_year": 17_520.0,
                }
            },
            "warnings": (),
        }
        context = DSEDesignContext(
            design_id="leo-context-that-must-not-leak",
            design_row={},
            runtime_scenario=SimpleNamespace(
                scenario_id="source", serving_deployments=(), provenance={}
            ),
            current_scheduler={"policy": "local_only"},
            cost_result={
                "total_cost_usd": 1_000_000.0,
                "annualized_cost_usd_per_year": 100_000.0,
            },
        )
        spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("skywalker_like_ground",),
            ground_config=_ground_config(include_egress=False),
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=True,
        )
        fake = SimpleNamespace(to_dict=lambda: payload)

        with patch(
            "orbitserve.dse.runtime_matrix.evaluate_runtime_performance",
            return_value=fake,
        ):
            result = evaluate_runtime_matrix((context,), spec).to_dict()

        row = result["matrix_rows"][0]
        self.assertEqual(row["design_id"], "terrestrial_reference")
        self.assertEqual(row["cost_total_usd"], 2.0)
        self.assertEqual(row["annualized_cost_usd_per_year"], 17_520.0)
        self.assertEqual(row["cost_per_served_request_usd"], 2.0)
        self.assertNotEqual(row["annualized_cost_usd_per_year"], 100_000.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
