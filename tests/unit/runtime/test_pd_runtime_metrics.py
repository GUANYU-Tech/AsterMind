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

        self.assertEqual(payload["assigned_satellite_id"], "sat-01")
        self.assertEqual(payload["compute_owner_satellite_id"], "sat-01")
        self.assertNotIn("prefill_deployment_id", payload)
        self.assertNotIn("decode_deployment_id", payload)
        self.assertFalse(any(key.startswith("pd_") for key in payload))
        self.assertFalse(any(key.startswith("kv_handoff") for key in payload))

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

        self.assertEqual(payload["prefill_service_s"], 0.5)
        self.assertEqual(payload["decode_service_s"], 2.0)


if __name__ == "__main__":
    unittest.main()
