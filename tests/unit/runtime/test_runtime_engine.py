import unittest

from orbitserve.orbit import ORBIT_TRACE_SCHEMA, OrbitTraceBundle, OrbitTraceManifest
from orbitserve.experiments.simulation_builder import build_simulation_spec
from orbitserve.runtime import RuntimeStrategySpec, run_simulation


def _trace():
    manifest = OrbitTraceManifest(
        schema_version=ORBIT_TRACE_SCHEMA,
        trace_model="unit_geometry",
        evaluated=True,
        trace_id="unit-trace",
        time={"epoch": "2026-01-01T00:00:00Z", "start_s": 0.0, "end_s": 100.0, "duration_s": 100.0, "step_s": 1.0},
        files={},
        sha256={},
        valid_for_paper_main_results=False,
        scope="orbit_geometry_only",
        warnings=(),
        provenance={},
    )
    return OrbitTraceBundle(
        trace_manifest=manifest,
        satellite_catalog=(
            {"satellite_id": "sat-a", "compute_capable": True},
            {"satellite_id": "sat-b", "compute_capable": True},
        ),
        ephemeris_samples=(
            {"time_s": 0.0, "satellite_id": "sat-a", "position_itrs_km": [7000.0, 0.0, 0.0]},
            {"time_s": 0.0, "satellite_id": "sat-b", "position_itrs_km": [0.0, 7000.0, 0.0]},
        ),
        sgl_windows=(),
        isl_windows=(
            {
                "link_id": "isl:sat-a:sat-b",
                "endpoint_a": "sat-a",
                "endpoint_b": "sat-b",
                "start_s": 0.0,
                "end_s": 100.0,
                "distance_km": 9000.0,
                "propagation_delay_s": 0.03,
            },
        ),
        sunlight_windows=(
            {"satellite_id": "sat-a", "start_s": 0.0, "end_s": 100.0, "sunlit": False},
            {"satellite_id": "sat-b", "start_s": 0.0, "end_s": 100.0, "sunlit": True},
        ),
    )


def _request(
    request_id="request-0",
    arrival_time_s=0.0,
    lon_deg=0.0,
    context_tokens_before_turn=0,
    session_id="session-0",
):
    return {
        "request_id": request_id,
        "arrival_time_s": arrival_time_s,
        "lat_deg": 0.0,
        "lon_deg": lon_deg,
        "ingress_id": "gateway-0",
        "prompt_tokens": 10,
        "output_tokens": 2,
        "input_bytes": 1000.0,
        "output_bytes": 200.0,
        "session_id": session_id,
        "context_tokens_before_turn": context_tokens_before_turn,
        "slo_class": "interactive",
    }


def _scenario(strategy, requests=None, trace=None):
    return build_simulation_spec(
        workload_generation={
            "requests": requests or [_request()]
        },
        constellation_architecture={
            "storage_capacity_tb": 2e-10,
            "communication": {
                "sgl_bandwidth_bps": 1e9,
                "isl_bandwidth_bps": 1e9,
            }
        },
        simulation_result=trace or _trace(),
        simulation_config={
            "communication_satellite": {
                "altitude_km": 550.0,
                "uplink_bandwidth_bps": 1e9,
                "downlink_bandwidth_bps": 1e9,
            }
        },
        serving_profile={
            "evaluated": True,
            "feasible": True,
            "prefill_time_table": [
                {"prompt_length": 1, "prefill_tokens_per_s": 10.0},
                {"prompt_length": 100, "prefill_tokens_per_s": 10.0},
            ],
            "decode_time_table": [
                {"prompt_length": 1, "decode_tokens_per_s": 10.0},
                {"prompt_length": 100, "decode_tokens_per_s": 10.0},
            ],
            "kv_cache_capacity_tokens": 100,
            "kv_cache_bytes_per_token": 2.0,
        },
        energy_profile={
            "evaluated": True,
            "battery_capacity_j": 1000.0,
            "charge_rate_w": 20.0,
            "idle_power_w": 1.0,
            "compute_energy_j_per_token": 1.0,
            "link_energy_j_per_byte": 1e-6,
            "soc_min": 0.1,
            "initial_soc": 0.8,
            "battery_charge_efficiency": 1.0,
            "battery_discharge_efficiency": 1.0,
        },
        strategy=strategy,
    )


class RuntimeEngineTests(unittest.TestCase):
    def test_energy_score_observes_solar_input_and_no_link_queue_exists(self):
        strategy = RuntimeStrategySpec(
            latency_weight=0.0,
            kv_reuse_weight=0.0,
            energy_weight=1.0,
            compute_load_weight=0.0,
        )
        result = run_simulation(_scenario(strategy))
        self.assertTrue(result.evaluated)
        record = result.request_records[0]
        self.assertEqual(record["decision"]["serving_satellite_id"], "sat-b")
        self.assertNotIn("links", result.state_summary)
        self.assertNotIn("link_queue_wait_s", record["timing"])
        self.assertFalse(result.slo_metrics["evaluated"])
        self.assertEqual(
            result.state_summary["kv_caches"]["sat-a"]["capacity_tokens"],
            100,
        )

    def test_ground_access_latency_contains_transmission_and_propagation(self):
        strategy = RuntimeStrategySpec(
            latency_weight=1.0,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.0,
        )
        result = run_simulation(_scenario(strategy))
        record = result.request_records[0]
        first_leg = record["routing"]["selected_path"][0]
        self.assertAlmostEqual(first_leg["propagation_delay_s"], 550.0 / 299_792.458)
        self.assertAlmostEqual(first_leg["transmission_s"], 1000.0 * 8.0 / 1e9)

    def test_shadow_dvfs_scales_compute_capacity(self):
        strategy = RuntimeStrategySpec(
            latency_weight=1.0,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.0,
            shadow_dvfs_enabled=True,
            shadow_dvfs_scale=0.5,
        )
        result = run_simulation(_scenario(strategy))
        record = result.request_records[0]
        self.assertEqual(record["decision"]["serving_satellite_id"], "sat-a")
        self.assertEqual(record["decision"]["dvfs_scale"], 0.5)
        self.assertEqual(record["timing"]["prefill_service_s"], 2.0)

    def test_equal_power_and_compute_dvfs_keeps_energy_per_token_constant(self):
        full_speed = run_simulation(
            _scenario(
                RuntimeStrategySpec(
                    latency_weight=1.0,
                    kv_reuse_weight=0.0,
                    energy_weight=0.0,
                    compute_load_weight=0.0,
                )
            )
        ).request_records[0]
        shadow_dvfs = run_simulation(
            _scenario(
                RuntimeStrategySpec(
                    latency_weight=1.0,
                    kv_reuse_weight=0.0,
                    energy_weight=0.0,
                    compute_load_weight=0.0,
                    shadow_dvfs_enabled=True,
                    shadow_dvfs_scale=0.5,
                )
            )
        ).request_records[0]
        self.assertEqual(
            full_speed["energy"]["compute_energy_j"],
            shadow_dvfs["energy"]["compute_energy_j"],
        )

    def test_full_kv_copy_reuses_context_and_retains_source(self):
        strategy = RuntimeStrategySpec(
            latency_weight=1.0,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.0,
            kv_cache_copy_enabled=True,
        )
        result = run_simulation(
            _scenario(
                strategy,
                requests=(
                    _request("request-a", 0.0, 0.0),
                    _request("request-b", 10.0, 90.0, 10),
                ),
            )
        )
        first, second = result.request_records
        self.assertEqual(first["decision"]["serving_satellite_id"], "sat-a")
        self.assertEqual(second["decision"]["serving_satellite_id"], "sat-b")
        self.assertEqual(first["decision"]["serving_compute_node_id"], "sat-a")
        self.assertEqual(second["decision"]["serving_compute_node_id"], "sat-b")
        self.assertEqual(second["timing"]["prefill_service_s"], 0.0)
        self.assertNotIn("kv_migrated_bytes", second["kv"])
        self.assertIn("kv_copy", {leg["kind"] for leg in second["routing"]["selected_path"]})
        self.assertIn("session-0", result.state_summary["kv_caches"]["sat-a"]["entries"])
        self.assertIn("session-0", result.state_summary["kv_caches"]["sat-b"]["entries"])
        self.assertIn("sat-a", result.state_summary["compute_nodes"])
        self.assertNotIn("satellites", result.state_summary)
        self.assertNotIn("satellite_ids", result.state_summary["sessions"]["session-0"])
        self.assertEqual(second["kv"]["kv_copied_tokens"], 10)
        self.assertGreater(
            result.state_summary["kv_caches"]["sat-a"]["entries"]["session-0"]["last_access_s"],
            10.0,
        )

    def test_disabled_kv_copy_recomputes_the_context(self):
        strategy = RuntimeStrategySpec(
            latency_weight=1.0,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.0,
            kv_cache_copy_enabled=False,
        )
        result = run_simulation(
            _scenario(
                strategy,
                requests=(
                    _request("request-a", 0.0, 0.0),
                    _request("request-b", 10.0, 90.0, 10),
                ),
            )
        )
        second = result.request_records[1]
        self.assertEqual(second["timing"]["prefill_service_s"], 1.0)
        self.assertEqual(second["kv"]["context_recompute_tokens"], 10)

    def test_kv_copy_never_reuses_across_sessions(self):
        strategy = RuntimeStrategySpec(
            latency_weight=1.0,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.0,
            kv_cache_copy_enabled=True,
        )
        result = run_simulation(
            _scenario(
                strategy,
                requests=(
                    _request("request-a", 0.0, 0.0, session_id="session-a"),
                    _request("request-b", 10.0, 90.0, 10, session_id="session-b"),
                ),
            )
        )
        second = result.request_records[1]
        self.assertEqual(second["kv"]["kv_copied_tokens"], 0)
        self.assertEqual(second["kv"]["context_recompute_tokens"], 10)

    def test_reuse_is_capped_by_declared_prior_context(self):
        strategy = RuntimeStrategySpec(
            latency_weight=1.0,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.0,
            kv_cache_copy_enabled=True,
        )
        result = run_simulation(
            _scenario(
                strategy,
                requests=(
                    _request("request-a", 0.0, 0.0),
                    _request("request-b", 10.0, 90.0, 5),
                ),
            )
        )
        second = result.request_records[1]
        self.assertEqual(second["kv"]["kv_copied_tokens"], 5)
        self.assertEqual(second["kv"]["context_recompute_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
