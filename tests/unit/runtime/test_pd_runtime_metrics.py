import unittest

from orbitserve.runtime.metrics.runtime_metrics import aggregate_runtime_records
from orbitserve.runtime.metrics.runtime_records import RuntimeRequestRecord, records_to_jsonable


def _record(request_id: str, **overrides):
    values = {
        "request_id": request_id,
        "arrival_time_s": 0.0,
        "deployment_id": "deployment-0",
        "action_type": "local_compute",
        "completed": True,
        "prompt_tokens": 128,
        "output_tokens": 4,
        "ttft_s": 1.0,
        "e2e_s": 4.0,
        "assigned_satellite_id": "sat-01",
        "compute_owner_satellite_id": "sat-01",
        "compute_owner_satellite_ids": ("sat-01",),
        "scope": "idealized_compute_only",
        "trace_model": "ideal_full_visibility",
    }
    values.update(overrides)
    return RuntimeRequestRecord(**values)


class SingleSatelliteRuntimeMetricsTests(unittest.TestCase):
    def test_record_identifies_one_full_serving_satellite(self):
        payload = records_to_jsonable((_record("req-0"),))[0]

        self.assertEqual(payload["decision"]["serving_satellite_id"], "sat-01")
        self.assertEqual(payload["placement"]["compute_owner_satellite_id"], "sat-01")
        serialized_keys = {key for section in payload.values() for key in section}
        self.assertNotIn("prefill_deployment_id", serialized_keys)
        self.assertNotIn("decode_deployment_id", serialized_keys)
        self.assertFalse(any(key.startswith("pd_") for key in serialized_keys))
        self.assertFalse(any(key.startswith("kv_handoff") for key in serialized_keys))

    def test_record_has_all_responsibility_sections(self):
        payload = records_to_jsonable((_record("req-sections"),))[0]

        self.assertEqual(
            tuple(payload),
            ("request", "decision", "placement", "timing", "routing", "energy", "kv", "outcome", "provenance"),
        )
        self.assertEqual(payload["request"]["request_id"], "req-sections")
        self.assertTrue(payload["outcome"]["completed"])

    def test_metrics_have_no_phase_separation_fields(self):
        metrics = aggregate_runtime_records(
            (_record("req-0"),),
            time_horizon_s=10.0,
            trace_model="ideal_full_visibility",
            scope="idealized_compute_only",
        ).runtime_metrics

        self.assertEqual(metrics["tpot_p50_s"], 1.0)
        self.assertFalse(any(key.startswith("pd_") for key in metrics))
        self.assertNotIn("kv_handoff_bytes_total", metrics)

    def test_prefill_and_decode_timings_remain_available(self):
        payload = records_to_jsonable(
            (_record("req-1", prefill_service_s=0.5, decode_service_s=2.0),)
        )[0]

        self.assertEqual(payload["timing"]["prefill_service_s"], 0.5)
        self.assertEqual(payload["timing"]["decode_service_s"], 2.0)

    def test_routing_and_energy_facts_are_derived_from_recorded_events(self):
        payload = records_to_jsonable((
            _record(
                "req-facts",
                selected_path=(
                    {"link_id": "sgl-1", "kind": "sgl", "propagation_delay_s": 0.1},
                    {"link_id": "isl-1", "kind": "isl", "propagation_delay_s": 0.2},
                ),
                energy_ledger=(
                    {"compute_bus_j": 4.0, "link_bus_j": 1.5},
                    {"compute_bus_j": 0.0, "link_bus_j": 0.5},
                ),
            ),
        ))[0]

        self.assertEqual(payload["routing"]["route_links"], ["sgl-1", "isl-1"])
        self.assertEqual(payload["routing"]["hop_count"], 1)
        self.assertAlmostEqual(payload["timing"]["propagation_s"], 0.3)
        self.assertEqual(payload["energy"]["compute_energy_j"], 4.0)
        self.assertEqual(payload["energy"]["communication_energy_j"], 2.0)


if __name__ == "__main__":
    unittest.main()
