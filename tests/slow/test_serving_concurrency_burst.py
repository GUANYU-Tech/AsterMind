import unittest

from orbitserve.runtime import RuntimeSimulationSpec, RuntimeStrategySpec, run_simulation
from tests.integration.fixed_design.test_serving_runtime_concurrency import _trace


class ServingConcurrencyBurstSlowTests(unittest.TestCase):
    def test_large_burst_never_exceeds_node_concurrency(self):
        lengths = (1, 100, 1000)
        max_concurrency = 8
        prefill = [
            {"prompt_length": length, "concurrency": concurrency, "prefill_tokens_per_s": 1000.0 / concurrency}
            for length in lengths for concurrency in range(1, max_concurrency + 1)
        ]
        decode = [
            {"context_length": length, "concurrency": concurrency, "decode_tokens_per_s": 500.0 / concurrency}
            for length in lengths for concurrency in range(1, max_concurrency + 1)
        ]
        requests = tuple({
            "request_id": f"request-{index:04d}",
            "arrival_time_s": 0.0,
            "prompt_tokens": 100,
            "output_tokens": 10,
            "input_bytes": 1000.0,
            "output_bytes": 1000.0,
            "session_id": f"session-{index:04d}",
        } for index in range(200))
        spec = RuntimeSimulationSpec(
            strategy=RuntimeStrategySpec(
                latency_weight=1.0, kv_reuse_weight=0.0,
                energy_weight=0.0, compute_load_weight=0.0,
            ),
            requests=requests,
            environment_state=_trace(),
            constellation={"architecture": {"storage_capacity_tb": 1.0}},
            serving_profile={
                "evaluated": True, "feasible": True,
                "prefill_rate_table": prefill, "decode_rate_table": decode,
                "max_concurrency": max_concurrency,
                "kv_cache_bytes_per_token": 2.0,
                "kv_cache_capacity_tokens": 1_000_000,
            },
            communication_ingress={
                "altitude_km": 550.0,
                "uplink_bandwidth_bps": 1e9,
                "downlink_bandwidth_bps": 1e9,
            },
            energy_profile={
                "battery_capacity_j": 1e12, "charge_rate_w": 1000.0,
                "idle_power_w": 0.0, "compute_energy_j_per_token": 0.0,
                "link_energy_j_per_byte": 0.0, "soc_min": 0.0,
                "initial_soc": 1.0,
            },
        )
        result = run_simulation(spec)
        self.assertEqual(result.request_summary["completed_request_count"], 200)
        reservations = result.state_summary["compute_nodes"]["sat-a"]["compute_queue"]["reservations"]
        for time_s in {row["start_s"] for row in reservations}:
            overlap = sum(
                row["start_s"] <= time_s < row["finish_s"]
                for row in reservations
            )
            self.assertLessEqual(overlap, max_concurrency)


if __name__ == "__main__":
    unittest.main()
