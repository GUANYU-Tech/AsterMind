import unittest

from orbitserve.architecture import HardwareSpec, resolve_architecture
from orbitserve.orbit import ORBIT_TRACE_SCHEMA, OrbitTraceBundle, OrbitTraceManifest
from orbitserve.runtime import RuntimeSimulationSpec, RuntimeStrategySpec, run_simulation, NetworkSpec, NetworkNodeSpec, LinkProfile
from orbitserve.serving import ServingSpec, estimate_serving_throughput


def _trace():
    manifest = OrbitTraceManifest(
        schema_version=ORBIT_TRACE_SCHEMA,
        trace_model="integration_geometry",
        evaluated=True,
        trace_id="serving-runtime-concurrency",
        time={"epoch": "2026-01-01T00:00:00Z", "start_s": 0.0, "end_s": 100.0, "duration_s": 100.0, "step_s": 1.0},
        files={}, sha256={}, valid_for_paper_main_results=False,
        scope="orbit_geometry_only", warnings=(), provenance={},
    )
    return OrbitTraceBundle(
        trace_manifest=manifest,
        satellite_catalog=({"satellite_id": "sat-a", "compute_capable": True},),
        ephemeris_samples=tuple({"time_s": t, "satellite_id": "sat-a", "position_itrs_km": [7000.0, 0.0, 0.0]} for t in (0.,100.)),
        sgl_windows=({"ingress_id":"ground","satellite_id":"sat-a","start_s":0.,"end_s":100.},), isl_windows=(),
        sunlight_windows=({"satellite_id": "sat-a", "start_s": 0.0, "end_s": 100.0, "sunlit": True},),
    )


class ServingRuntimeConcurrencyIntegrationTests(unittest.TestCase):
    def test_serving_rate_tables_drive_runtime_admission_slots(self):
        hardware = resolve_architecture(HardwareSpec.from_dict({
            "accelerator": "h200_sxm",
            "compute_payload": {"accelerator_count": 4, "purchase_cost_per_unit_usd": 50000},
        })).hardware_profile
        serving = estimate_serving_throughput(
            ServingSpec.from_dict({
                "max_concurrency": 2,
                "prompt_length_grid": [1, 10, 100],
                "context_length_grid": [1, 10, 100],
            }),
            hardware=hardware,
        )
        requests = tuple({
            "request_id": f"request-{index}", "ingress_id":"ground",
            "arrival_time_s": 0.0,
            "prompt_tokens": 10,
            "output_tokens": 2,
            "input_bytes": 1000.0,
            "output_bytes": 200.0,
            "session_id": f"session-{index}",
        } for index in range(3))
        spec = RuntimeSimulationSpec(
            strategy=RuntimeStrategySpec(
                latency_weight=1.0, kv_reuse_weight=0.0,
                energy_weight=0.0, compute_load_weight=0.0,
            ),
            requests=requests,
            environment_state=_trace(),
            constellation={"architecture": {"storage_capacity_tb": 1.0}},
            serving_profile=serving,
            network=NetworkSpec((NetworkNodeSpec("sat-a","compute",sgl_terminals=1),
                NetworkNodeSpec("ground","ground",sgl_terminals=1)),
                {"sgl":LinkProfile(1e9,0.)},communication_topology="explicit",ground_roles=("compute",)),
            gateways=({"ingress_id":"ground","lat_deg":0.,"lon_deg":0.},),
            energy_profile={
                "battery_capacity_j": 1e9, "charge_rate_w": 1000.0,
                "idle_power_w": 1.0, "compute_energy_j_per_token": 0.0,
                "link_energy_j_per_byte": 0.0, "soc_min": 0.1,
                "initial_soc": 0.8,
            },
        )
        result = run_simulation(spec)
        self.assertTrue(result.evaluated)
        self.assertEqual(
            [row["provenance"]["serving_concurrency_at_admission"] for row in result.request_records],
            [1, 2, 2],
        )
        self.assertGreater(result.request_records[2]["timing"]["prefill_queue_wait_s"], 0.0)
        self.assertEqual(set(result.inference_metrics), {"ttft_s", "tpot_s", "e2e_s"})


if __name__ == "__main__":
    unittest.main()
