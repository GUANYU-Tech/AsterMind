"""Transport arrivals retain exact time, acquisition delay and horizon failures."""
import unittest
from dataclasses import replace
from orbitserve.runtime import LinkProfile
from tests.network_case import NetworkCase, traffic, run_traffic

class NetworkRequestEventTests(unittest.TestCase):
    def test_acquisition_delays_transfer_without_rounding_to_topology_step(self):
        spec = NetworkCase(traffic=(traffic("between-samples", 0.25),))
        network = replace(spec.network, link_profiles={**spec.network.link_profiles, "sgl": LinkProfile(1e9, 3.0)})
        result = run_traffic(replace(spec, network=network))
        self.assertEqual(result.completed, 1)
        first = result.transfer_jobs[0]["path"][0]
        self.assertEqual(first["start_s"], 3.0)
        self.assertTrue({"transmit_start", "hop_arrival"} <= {e["kind"] for e in result.log})

    def test_horizon_and_missing_ingress_never_report_completed_transfers(self):
        for arrival, ingress in [(101.0, "gateway-0"), (2.0, "unknown")]:
            value = {**traffic("bad", arrival), "ingress_id": ingress}
            result = run_traffic(NetworkCase(traffic=(value,)))
            self.assertEqual(result.completed, 0)
            self.assertNotIn("completed_at", result.transfer_jobs[0])
