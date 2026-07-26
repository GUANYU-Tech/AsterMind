"""Unit tests for orbitserve.workload.upsample.stream_upsampled_sessions."""

import math
import types
import unittest

from orbitserve.workload import (
    REQUEST_KEYS,
    SESSION_KEYS,
    WorkloadGenerationResult,
    RegionSpec,
)
from orbitserve.workload.upsample import stream_upsampled_sessions


BIN_S = 300.0
HORIZON = 86400.0


def _request(request_id, session_id, turn, arrival, turn_count, gap=None):
    return {
        "request_id": request_id,
        "arrival_time_s": float(arrival),
        "user_region_id": "r1",
        "region_id": "r1",
        "lat_deg": 0.0,
        "lon_deg": 0.0,
        "access_mode": "direct_user_sgl",
        "ingress_id": "r1",
        "ingress_lat_deg": 0.0,
        "ingress_lon_deg": 0.0,
        "gateway_id": None,
        "task_type": "conversation",
        "model_family": "m",
        "prompt_tokens": 10 + turn,
        "new_user_tokens": 10,
        "context_tokens_before_turn": turn,
        "output_tokens": 5,
        "input_bytes": 100,
        "output_bytes": 50,
        "context_window_tokens": 1024,
        "slo_class": "interactive",
        "ttft_deadline_s": 1.0,
        "e2e_deadline_s": 5.0,
        "priority": "interactive",
        "session_id": session_id,
        "turn_index": turn,
        "session_turn_count": turn_count,
        "inter_turn_gap_s": gap,
        "session_ttl_s": 600.0,
        "burst_id": None,
        "burst_multiplier": 1.0,
        "seed": 1,
    }


def _workload():
    sessions = [
        {"session_id": "s1", "user_region_id": "r1", "session_start_time_s": 10.0, "session_turn_count": 3,
         "session_slo_class": "interactive", "context_window_tokens": 1024, "session_ttl_s": 600.0, "seed": 11},
        {"session_id": "s2", "user_region_id": "r1", "session_start_time_s": 400.0, "session_turn_count": 2,
         "session_slo_class": "interactive", "context_window_tokens": 1024, "session_ttl_s": 600.0, "seed": 22},
    ]
    requests = [
        _request("r1a", "s1", 0, 10.0, 3),
        _request("r1b", "s1", 1, 15.0, 3, gap=5.0),
        _request("r1c", "s1", 2, 25.0, 3, gap=10.0),
        _request("r2a", "s2", 0, 400.0, 2),
        _request("r2b", "s2", 1, 450.0, 2, gap=50.0),
    ]
    manifest = {
        "workload_id": "wl_test",
        "seed": 7,
        "time": {"epoch": "2023-05-15T00:00:00Z", "time_horizon_s": HORIZON, "time_bin_s": BIN_S},
    }
    region = RegionSpec.from_dict({"region_id": "r1", "lat_deg": 0.0, "lon_deg": 0.0})
    return WorkloadGenerationResult(
        bundle_dir=None,
        manifest=manifest,
        regions=(region,),
        gateways=(),
        requests=tuple(requests),
        sessions=tuple(sessions),
        timebin_demand=(),
        summary={},
    )


def _signature(stream):
    return [
        (session, [tuple(sorted(row.items())) for row in rows])
        for session, rows in stream
    ]


class UpsampleTests(unittest.TestCase):
    def test_factor_one_is_identity(self):
        workload = _workload()
        out = list(stream_upsampled_sessions(workload, 1, seed="fixed"))
        self.assertEqual(len(out), 2)
        for (session, rows), (orig_session, orig_rows) in zip(
            out,
            [(workload.sessions[0], list(workload.requests[:3])), (workload.sessions[1], list(workload.requests[3:]))],
        ):
            self.assertEqual(session, dict(orig_session))
            self.assertEqual(rows, [dict(row) for row in orig_rows])

    def test_is_lazy_generator(self):
        stream = stream_upsampled_sessions(_workload(), 1000, seed="lazy")
        self.assertIsInstance(stream, types.GeneratorType)
        first = next(stream)
        self.assertEqual(first[0]["session_id"].startswith("s1_up0_"), True)

    def test_factor_counts_are_exact(self):
        out = list(stream_upsampled_sessions(_workload(), 5, seed="n"))
        self.assertEqual(len(out), 2 * 5)
        self.assertEqual(sum(len(rows) for _, rows in out), 5 * (3 + 2))
        per_base = {}
        for session, rows in out:
            self.assertTrue(all(row["session_id"] == session["session_id"] for row in rows))
            per_base.setdefault(session["session_id"].split("_up")[0], 0)
            per_base[session["session_id"].split("_up")[0]] += 1
        self.assertEqual(per_base, {"s1": 5, "s2": 5})

    def test_deterministic_across_runs(self):
        first = _signature(stream_upsampled_sessions(_workload(), 4, seed="abc"))
        second = _signature(stream_upsampled_sessions(_workload(), 4, seed="abc"))
        self.assertEqual(first, second)

    def test_ids_are_unique_and_rederived(self):
        out = list(stream_upsampled_sessions(_workload(), 4, seed="ids"))
        session_ids = [session["session_id"] for session, _ in out]
        self.assertEqual(len(session_ids), len(set(session_ids)))
        request_ids = [row["request_id"] for _, rows in out for row in rows]
        self.assertEqual(len(request_ids), len(set(request_ids)))
        self.assertFalse(set(session_ids) & {"s1", "s2"})
        self.assertFalse(set(request_ids) & {"r1a", "r1b", "r1c", "r2a", "r2b"})
        # seed 变化则派生 id 变化
        other = list(stream_upsampled_sessions(_workload(), 4, seed="ids2"))
        other_session_ids = [session["session_id"] for session, _ in other]
        self.assertFalse(set(session_ids) & set(other_session_ids))

    def test_jitter_stays_in_bin_and_horizon(self):
        workload = _workload()
        original_first = {"s1": 10.0, "s2": 400.0}
        out = list(stream_upsampled_sessions(workload, 20, seed="jitter"))
        for session, rows in out:
            base = session["session_id"].split("_up")[0]
            arrivals = [row["arrival_time_s"] for row in rows]
            self.assertTrue(arrivals)
            self.assertEqual(math.floor(arrivals[0] / BIN_S), math.floor(original_first[base] / BIN_S))
            self.assertTrue(all(0.0 <= arrival < HORIZON for arrival in arrivals))
            self.assertEqual(math.floor(session["session_start_time_s"] / BIN_S), math.floor(original_first[base] / BIN_S))

    def test_turn_structure_is_preserved(self):
        workload = _workload()
        originals = {}
        for row in workload.requests:
            originals.setdefault(row["session_id"], []).append(row)
        out = list(stream_upsampled_sessions(workload, 3, seed="turns"))
        for session, rows in out:
            base = session["session_id"].split("_up")[0]
            base_rows = sorted(originals[base], key=lambda row: row["turn_index"])
            rows = sorted(rows, key=lambda row: row["turn_index"])
            self.assertEqual([row["turn_index"] for row in rows], [row["turn_index"] for row in base_rows])
            self.assertEqual([row["session_turn_count"] for row in rows], [row["session_turn_count"] for row in base_rows])
            self.assertEqual([row["inter_turn_gap_s"] for row in rows], [row["inter_turn_gap_s"] for row in base_rows])
            self.assertEqual(session["session_turn_count"], len(rows))
            gaps = [b["arrival_time_s"] - a["arrival_time_s"] for a, b in zip(rows, rows[1:])]
            base_gaps = [b["arrival_time_s"] - a["arrival_time_s"] for a, b in zip(base_rows, base_rows[1:])]
            self.assertEqual(gaps, base_gaps)

    def test_request_and_session_field_sets(self):
        out = list(stream_upsampled_sessions(_workload(), 2, seed="keys"))
        for session, rows in out:
            self.assertEqual(set(session), set(SESSION_KEYS))
            for row in rows:
                self.assertEqual(set(row), set(REQUEST_KEYS))

    def test_factor_validation(self):
        for bad in (0, -2, 1.5, "3", None):
            with self.assertRaises(ValueError, msg=f"factor={bad!r}"):
                list(stream_upsampled_sessions(_workload(), bad, seed="x"))


if __name__ == "__main__":
    unittest.main()
