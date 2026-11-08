import copy
import json
import unittest
from pathlib import Path

from simulator.config import SimulationConfig
from simulator.replay import audit_checks, native_ns, replay_data


def report():
    """Small recorded-contract fixture; actual native runs are checked separately."""
    path = Path(__file__).resolve().parents[3] / "simulator/examples/two_cells.json"
    config = json.loads(path.read_text())
    config["requests"] = [config["requests"][0]]
    config["requests"][0]["output_tokens"] = 1
    graph = SimulationConfig.model_validate(config).topology()
    call = {
        "call_id": "qkv",
        "occurrence": -1,
        "rank": 0,
        "kind": "GEMM",
        "communication": False,
        "dependencies": [],
        "start_time_ns": 2,
        "end_time_ns": 5,
        "phases": [],
        "native_prediction_s": 3e-9,
        "host_ready_time_ns": 1,
        "launch_gap_ns": 0,
    }
    flow = {
        "id": "flow",
        "source": "gpu0",
        "target": "gpu1",
        "size_bytes": 100,
        "ready_time_ns": 5,
        "operation": "pp_send_recv",
        "correlation_id": "cellA/main:0",
    }
    comm = {
        "call_id": "pp",
        "occurrence": -1,
        "rank": 0,
        "kind": "COMMUNICATION",
        "communication": True,
        "dependencies": [["qkv", -1]],
        "start_time_ns": 5,
        "end_time_ns": 10,
        "phases": [[flow]],
        "native_prediction_s": 1e-9,
        "host_ready_time_ns": 1,
        "launch_gap_ns": 0,
    }
    event = {
        **flow,
        "kind": "completed",
        "completion_time_ns": 10,
        "path": ["gpu0", "switch0", "isl0", "isl1", "switch1", "gpu1"],
    }
    storage = {
        "capacity": {
            "source_mode": "debug_explicit_block_count",
            "num_kv_blocks": 64,
            "kv_block_bytes": 16,
            "rank_geometry": [],
        },
        "ready_cached_blocks": 0,
        "migration_reserved_blocks": 0,
    }
    cells = {
        c["id"]: {
            "steps": [],
            "final_load": {"storage": copy.deepcopy(storage)},
            "scheduler_runtime": {"version": "0.31.0"},
        }
        for c in config["cells"]
    }
    return {
        "configuration": config,
        "status": "completed",
        "requests": {
            "r0": {
                "status": "completed",
                "arrival_ns": 0,
                "cell_ready_ns": 1,
                "compute_complete_ns": 10,
                "completed_ns": 20,
                "received_tokens": 1,
                "cell": "cellA",
            }
        },
        "produced_outputs": [
            {
                "request_id": "r0",
                "token_count": 1,
                "token_start": 0,
                "produced_time_s": 1e-8,
            }
        ],
        "batches": [
            {
                "cell_id": "cellA",
                "step_id": "main:0",
                "start_time_ns": 2,
                "end_time_ns": 10,
                "native_communication_duration_charged_s": 0,
                "calls": [call, comm],
            }
        ],
        "cells": cells,
        "transfers": [event],
        "input_distributions": [],
        "migrations": {},
        "provenance": {},
        "network": {
            "device_topology": graph.view("devices"),
            "event_log": [],
            "flows": [{**event, "completion_time_ns": 9.2}],
        },
    }


class NativeReplayTests(unittest.TestCase):
    def test_recorded_contract_passes_and_preserves_exact_operator_times(self):
        value = report()
        self.assertTrue(all(c["state"] == "pass" for c in audit_checks(value)))
        data = replay_data(
            value, source_name="actual-report.json", source_sha256="a" * 64
        )
        self.assertEqual(data["calls"][1]["deps"], [0])
        self.assertEqual(data["calls"][0]["start"], 2)
        self.assertEqual(data["calls"][0]["end"], 5)
        self.assertEqual(data["transfers"][0]["physical_end"], 9.2)
        self.assertEqual(data["source"]["sha256"], "a" * 64)

    def test_operator_released_before_dependency_fails_with_evidence(self):
        value = report()
        value["batches"][0]["calls"][1]["start_time_ns"] = 4
        checks = {c["id"]: c for c in audit_checks(value)}
        self.assertEqual(checks["dependency_order"]["state"], "fail")
        self.assertIn(
            "batches[0].calls[1]",
            [v["evidence"] for v in checks["dependency_order"]["violations"]],
        )

    def test_native_duration_and_duplicate_communication_charge_are_detected(self):
        value = report()
        value["batches"][0]["calls"][0]["end_time_ns"] = 6
        value["batches"][0]["native_communication_duration_charged_s"] = 0.01
        checks = {c["id"]: c for c in audit_checks(value)}
        self.assertEqual(checks["kernel_duration"]["state"], "fail")
        self.assertEqual(checks["communication_accounting"]["state"], "fail")

    def test_missing_flow_and_changed_byte_count_cannot_pass_a_phase(self):
        for mutation in ("missing", "bytes"):
            value = report()
            if mutation == "missing":
                value["transfers"] = []
            else:
                value["transfers"][0]["size_bytes"] = 99
            checks = {c["id"]: c for c in audit_checks(value)}
            self.assertEqual(checks["communication_barrier"]["state"], "fail")

    def test_notification_before_physical_delivery_is_detected(self):
        value = report()
        value["network"]["flows"][0]["completion_time_ns"] = 10.5
        checks = {c["id"]: c for c in audit_checks(value)}
        self.assertEqual(checks["delivery_clock"]["state"], "fail")

    def test_service_rate_above_recorded_capacity_is_detected(self):
        value = report()
        value["network"]["event_log"] = [
            {
                "kind": "bandwidth_update",
                "allocated_bandwidth_bps": 200.0,
                "link_capacity_bps": 100.0,
                "bytes_sent": 1.0,
            }
        ]
        checks = {c["id"]: c for c in audit_checks(value)}
        self.assertEqual(checks["service_capacity"]["state"], "fail")

    def test_failed_migration_is_rollback_check_not_successful_delivery(self):
        value = report()
        failure = {
            "id": "migration",
            "kind": "failed",
            "source": "gpu0",
            "target": "gpu1",
            "size_bytes": 32,
            "path": ["gpu0", "switch0"],
            "failure_time_ns": 12,
            "operation": "kv_migration",
        }
        value["transfers"].append(failure)
        value["migrations"]["m"] = {
            "id": "m",
            "source": "cellA",
            "target": "cellB",
            "state": "failed",
            "bytes": 32,
            "start_time_s": 1e-9,
            "completed_time_s": 12e-9,
            "failures": [failure],
            "partitions": [
                {
                    "rank": 0,
                    "source": "gpu0",
                    "target": "gpu1",
                    "bytes": 32,
                    "transfer_id": "migration",
                }
            ],
        }
        checks = {c["id"]: c for c in audit_checks(value)}
        self.assertEqual(checks["migration_transaction"]["state"], "pass")
        value["cells"]["cellB"]["final_load"]["storage"][
            "migration_reserved_blocks"
        ] = 1
        checks = {c["id"]: c for c in audit_checks(value)}
        self.assertEqual(checks["migration_transaction"]["state"], "fail")

    def test_native_rounding_retains_sub_nanosecond_predictions(self):
        self.assertEqual(native_ns(1.00001e-9), 2)


if __name__ == "__main__":
    unittest.main()
