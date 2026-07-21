import sys
import unittest
from pathlib import Path


from orbitserve.runtime.metrics.runtime_metrics import aggregate_runtime_records
from orbitserve.runtime.metrics.runtime_records import RuntimeRequestRecord, records_to_jsonable


def _record(request_id: str, **overrides):
    values = {
        "request_id": request_id,
        "arrival_time_s": 0.0,
        "deployment_id": "deployment-0",
        "action_type": "local_compute",
        "accepted": True,
        "completed": True,
        "served": True,
        "rejected": False,
        "timeout": False,
        "prompt_tokens": 128,
        "output_tokens": 32,
        "ttft_s": 1.0,
        "e2e_s": 4.0,
        "scope": "idealized_compute_only",
        "trace_model": "ideal_full_visibility",
    }
    values.update(overrides)
    return RuntimeRequestRecord(**values)


class PDRuntimeMetricsTests(unittest.TestCase):
    def test_tpot_uses_inter_token_intervals_after_first_token(self):
        multi = aggregate_runtime_records(
            (_record("multi", output_tokens=4, ttft_s=1.0, e2e_s=4.0),),
            time_horizon_s=10.0,
            trace_model="ideal_full_visibility",
            scope="idealized_compute_only",
        ).runtime_metrics
        single = aggregate_runtime_records(
            (_record("single", output_tokens=1, ttft_s=1.0, e2e_s=1.0),),
            time_horizon_s=10.0,
            trace_model="ideal_full_visibility",
            scope="idealized_compute_only",
        ).runtime_metrics

        self.assertEqual(multi["tpot_p50_s"], 1.0)
        self.assertIsNone(single["tpot_p50_s"])

    def test_records_to_jsonable_includes_pd_runtime_fields(self):
        record = _record(
            "req-pd",
            prefill_deployment_id="prefill-0",
            decode_deployment_id="decode-0",
            prefill_entry_satellite_id="sat-prefill-entry",
            decode_entry_satellite_id="sat-decode-entry",
            pd_placement_scope="intra_cluster",
            resolved_pd_pair_scope="intra_cluster",
            pd_pairing_rule="user_explicit",
            kv_handoff_bytes=4096.0,
            kv_handoff_queue_wait_s=0.1,
            kv_handoff_transmission_s=0.2,
            kv_handoff_propagation_s=0.03,
            kv_handoff_setup_s=0.04,
            kv_handoff_visible_s=0.37,
            first_token_return_s=0.05,
            session_kv_migration_bytes=1024.0,
            session_kv_visible_s=0.01,
            kv_capacity_admission="pass",
            response_streaming_model="bulk_response_surrogate",
        )

        payload = records_to_jsonable((record,))[0]

        self.assertEqual(payload["prefill_deployment_id"], "prefill-0")
        self.assertEqual(payload["decode_deployment_id"], "decode-0")
        self.assertEqual(payload["prefill_entry_satellite_id"], "sat-prefill-entry")
        self.assertEqual(payload["decode_entry_satellite_id"], "sat-decode-entry")
        self.assertEqual(payload["pd_placement_scope"], "intra_cluster")
        self.assertEqual(payload["resolved_pd_pair_scope"], "intra_cluster")
        self.assertEqual(payload["pd_pairing_rule"], "user_explicit")
        self.assertEqual(payload["kv_handoff_bytes"], 4096.0)
        self.assertEqual(payload["kv_handoff_queue_wait_s"], 0.1)
        self.assertEqual(payload["kv_handoff_transmission_s"], 0.2)
        self.assertEqual(payload["kv_handoff_propagation_s"], 0.03)
        self.assertEqual(payload["kv_handoff_setup_s"], 0.04)
        self.assertEqual(payload["kv_handoff_visible_s"], 0.37)
        self.assertEqual(payload["first_token_return_s"], 0.05)
        self.assertEqual(payload["session_kv_migration_bytes"], 1024.0)
        self.assertEqual(payload["session_kv_visible_s"], 0.01)
        self.assertEqual(payload["kv_capacity_admission"], "pass")
        self.assertEqual(payload["response_streaming_model"], "bulk_response_surrogate")

    def test_runtime_metrics_aggregate_pd_kv_handoff_records(self):
        records = (
            _record(
                "req-0",
                kv_handoff_bytes=1024.0,
                kv_handoff_visible_s=0.5,
                pd_placement_scope="hybrid_template_mix",
                resolved_pd_pair_scope="intra_cluster",
                pd_pairing_rule="hybrid_template:same_plane_plus_cross_plane_overflow",
                session_action="sticky_kv_owner",
                kv_capacity_admission="pass",
            ),
            _record(
                "req-1",
                arrival_time_s=1.0,
                kv_handoff_bytes=2048.0,
                kv_handoff_queue_wait_s=0.2,
                kv_handoff_setup_s=0.1,
                kv_handoff_transmission_s=1.0,
                kv_handoff_propagation_s=0.2,
                pd_placement_scope="hybrid_template_mix",
                resolved_pd_pair_scope="same_plane_cross_cluster",
                pd_pairing_rule="hybrid_template:same_plane_plus_cross_plane_overflow",
                session_action="recompute_context",
                kv_capacity_admission="reject",
                provenance={"session": {"context_recompute_tokens": 64}},
            ),
        )

        aggregate = aggregate_runtime_records(
            records,
            time_horizon_s=10.0,
            trace_model="ideal_full_visibility",
            scope="idealized_compute_only",
        ).to_dict()
        metrics = aggregate["runtime_metrics"]

        self.assertEqual(metrics["pd_kv_handoff_bytes_total"], 3072.0)
        self.assertAlmostEqual(metrics["pd_kv_handoff_p95_s"], 1.45)
        self.assertEqual(metrics["pd_intra_cluster_request_fraction"], 0.5)
        self.assertEqual(metrics["pd_same_plane_request_fraction"], 0.5)
        self.assertEqual(metrics["pd_cross_plane_request_fraction"], 0.0)
        self.assertEqual(metrics["pd_pairing_rule_counts"], {"hybrid_template:same_plane_plus_cross_plane_overflow": 2})
        self.assertEqual(metrics["sticky_kv_owner_requests"], 1)
        self.assertEqual(metrics["recompute_context_requests"], 1)
        self.assertEqual(metrics["migrate_kv_requests"], 0)
        self.assertEqual(metrics["replicate_kv_requests"], 0)
        self.assertEqual(metrics["context_recompute_tokens_total"], 64.0)
        self.assertEqual(metrics["kv_capacity_reject_count"], 1)

    def test_runtime_metrics_non_pd_records_do_not_require_pd_fields(self):
        aggregate = aggregate_runtime_records(
            (_record("req-non-pd"),),
            time_horizon_s=10.0,
            trace_model="ideal_full_visibility",
            scope="idealized_compute_only",
        ).to_dict()
        metrics = aggregate["runtime_metrics"]

        self.assertEqual(metrics["pd_kv_handoff_bytes_total"], 0.0)
        self.assertIsNone(metrics["pd_kv_handoff_p95_s"])
        self.assertEqual(metrics["pd_intra_cluster_request_fraction"], 0.0)
        self.assertEqual(metrics["pd_same_plane_request_fraction"], 0.0)
        self.assertEqual(metrics["pd_cross_plane_request_fraction"], 0.0)
        self.assertEqual(metrics["kv_capacity_reject_count"], 0)
        self.assertEqual(metrics["migrate_kv_requests"], 0)
        self.assertEqual(metrics["replicate_kv_requests"], 0)

    def test_runtime_metrics_aggregate_kv_hit_transfer_and_replication_fields(self):
        records = (
            _record(
                "req-0",
                kv_serve_action="sticky_kv_owner",
                kv_post_turn_action="replicate_kv",
                kv_hit_tokens=80,
                kv_miss_tokens=20,
                kv_pages_required=10,
                kv_pages_hit=8,
                kv_reused_bytes=8000.0,
                kv_generated_bytes=2000.0,
                kv_migration_bytes=0.0,
                kv_replication_bytes=1000.0,
                kv_transfer_attempted_bytes=4000.0,
                kv_transfer_useful_bytes=3000.0,
                hbm_occupancy_after=0.5,
                context_recompute_tokens=0,
            ),
            _record(
                "req-1",
                kv_serve_action="migrate_kv",
                kv_post_turn_action="none",
                kv_hit_tokens=40,
                kv_miss_tokens=10,
                kv_pages_required=5,
                kv_pages_hit=4,
                kv_reused_bytes=4000.0,
                kv_generated_bytes=1000.0,
                kv_migration_bytes=4000.0,
                kv_transfer_attempted_bytes=6000.0,
                kv_transfer_useful_bytes=5000.0,
                kv_transfer_interrupted=True,
                hbm_occupancy_after=0.75,
                context_recompute_tokens=10,
            ),
        )

        metrics = aggregate_runtime_records(
            records,
            time_horizon_s=10.0,
            trace_model="ideal_full_visibility",
            scope="idealized_compute_only",
        ).to_dict()["runtime_metrics"]

        self.assertEqual(metrics["sticky_kv_owner_requests"], 1)
        self.assertEqual(metrics["migrate_kv_requests"], 1)
        self.assertEqual(metrics["replicate_kv_requests"], 1)
        self.assertEqual(metrics["kv_hit_rate_tokens"], 120 / 150)
        self.assertEqual(metrics["kv_hit_rate_pages"], 12 / 15)
        self.assertEqual(metrics["kv_reuse_bytes_total"], 12000.0)
        self.assertEqual(metrics["kv_migration_bytes_total"], 4000.0)
        self.assertEqual(metrics["kv_replication_bytes_total"], 1000.0)
        self.assertEqual(metrics["kv_transfer_waste_ratio"], 0.2)
        self.assertEqual(metrics["kv_transfer_interruption_count"], 1)
        self.assertEqual(metrics["post_turn_replication_request_fraction"], 0.5)
        self.assertEqual(metrics["context_recompute_tokens_total"], 10.0)
        self.assertEqual(metrics["hbm_occupancy_max"], 0.75)


if __name__ == "__main__":
    unittest.main()
