import unittest
from dataclasses import replace
from orbitserve.runtime import run_simulation, RuntimeStrategySpec, LinkProfile
from tests.unit.runtime.test_runtime_engine import _scenario, _request


class NetworkRequestEventTests(unittest.TestCase):
    def test_acquisition_delays_precise_request_without_rounding_to_topology_step(self):
        spec = _scenario(
            RuntimeStrategySpec(), requests=(_request("between-samples", 0.25),)
        )
        network = replace(
            spec.network,
            link_profiles={**spec.network.link_profiles, "sgl": LinkProfile(1e9, 3.0)},
        )
        result = run_simulation(replace(spec, network=network))
        row = result.request_records[0]
        self.assertTrue(row["outcome"]["completed"])
        self.assertGreaterEqual(row["provenance"]["compute_start_s"], 3.0)
        self.assertLess(row["provenance"]["compute_start_s"], 4.0)
        first = row["routing"]["selected_path"][0]
        self.assertEqual(first["start_s"], 3.0)
        kinds = {e["kind"] for e in result.state_summary["event_log"]}
        self.assertTrue(
            {
                "request_arrival",
                "transmit_start",
                "hop_arrival",
                "prefill_complete",
                "first_token_ready",
                "kv_publish",
                "request_complete",
            }
            <= kinds
        )

    def test_horizon_and_missing_ingress_never_report_completed_requests(self):
        for arrival, ingress in [(101.0, "gateway-0"), (2.0, "unknown")]:
            request = _request("bad", arrival)
            request["ingress_id"] = ingress
            result = run_simulation(
                _scenario(RuntimeStrategySpec(), requests=(request,))
            )
            self.assertFalse(result.request_records[0]["outcome"]["completed"])
            self.assertIsNone(result.request_records[0]["timing"]["e2e_s"])
