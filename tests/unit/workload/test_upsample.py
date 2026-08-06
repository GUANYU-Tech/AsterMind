"""Unit tests for orbitserve.workload.upsample.stream_upsampled_sessions."""
import math
import types
import unittest

from orbitserve.workload import (
    SESSION_REQUEST_KEYS,
    WorkloadGenerationResult,
    RegionSpec,
)
from orbitserve.workload.upsample import stream_upsampled_sessions


BIN_S = 300.0
HORIZON = 86400.0


def _request(request_id, session_id, turn, arrival, turn_count, gap=None,
             session_start=None):
    return {
        "request_id": request_id,
        "arrival_time_s": float(arrival),
        "user_region_id": "r1",
        "access_mode": "direct_user_sgl",
        "ingress_id": "r1",
        "gateway_id": None,
        "model_family": "chatgpt",
        "prompt_tokens": 10 + turn,
        "new_user_tokens": 10,
        "context_tokens_before_turn": turn,
        "output_tokens": 5,
        "input_bytes": 100,
        "output_bytes": 50,
        "context_window_tokens": 1024,
        "slo_class": "interactive_short",
        "ttft_deadline_s": 1.0,
        "e2e_deadline_s": 5.0,
        "session_id": session_id,
        "turn_index": turn,
        "session_turn_count": turn_count,
        "inter_turn_gap_s": gap,
        "session_start_time_s": session_start if session_start is not None else float(arrival),
        "session_ttl_s": 600.0,
        "seed": 1,
    }


def _workload():
    requests = [
        _request("r1a", "s1", 0, 10.0, 3, session_start=10.0),
        _request("r1b", "s1", 1, 15.0, 3, gap=5.0, session_start=10.0),
        _request("r1c", "s1", 2, 25.0, 3, gap=10.0, session_start=10.0),
        _request("r2a", "s2", 0, 400.0, 2, session_start=400.0),
        _request("r2b", "s2", 1, 450.0, 2, gap=50.0, session_start=400.0),
    ]
    manifest = {
        "workload_id": "wl_test",
        "seed": 7,
        "time_horizon_s": HORIZON,
    }
    region = RegionSpec.from_dict({"region_id": "r1", "region_name": "r1", "lat_deg": 0.0, "lon_deg": 0.0, "demand_weight": 1.0})
    return WorkloadGenerationResult(
        bundle_dir=None,
        manifest=manifest,
        regions=(region,),
        gateways=(),
        session_requests=tuple(requests),
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
        for (session, rows), orig_rows in zip(
            out,
            [list(workload.session_requests[:3]), list(workload.session_requests[3:])],
        ):
            self.assertIn(session["session_id"], {"s1", "s2"})
            self.assertEqual(len(rows), len(orig_rows))
            for r, o in zip(rows, orig_rows):
                self.assertEqual(r["request_id"], o["request_id"])
                self.assertEqual(r["turn_index"], o["turn_index"])

    def test_is_lazy_generator(self):
        stream = stream_upsampled_sessions(_workload(), 1000, seed="lazy")
        self.assertIsInstance(stream, types.GeneratorType)
        first = next(stream)
        self.assertTrue(first[0]["session_id"].startswith("s"))

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
        other = list(stream_upsampled_sessions(_workload(), 4, seed="ids2"))
        other_sids = [s["session_id"] for s, _ in other]
        self.assertFalse(set(session_ids) & set(other_sids))

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
        for row in workload.session_requests:
            originals.setdefault(row["session_id"], []).append(row)
        out = list(stream_upsampled_sessions(workload, 3, seed="turns"))
        for session, rows in out:
            base = session["session_id"].split("_up")[0]
            base_rows = sorted(originals[base], key=lambda r: r["turn_index"])
            rows = sorted(rows, key=lambda r: r["turn_index"])
            self.assertEqual([r["turn_index"] for r in rows], [r["turn_index"] for r in base_rows])
            self.assertEqual([r["session_turn_count"] for r in rows], [r["session_turn_count"] for r in base_rows])
            self.assertEqual([r["inter_turn_gap_s"] for r in rows], [r["inter_turn_gap_s"] for r in base_rows])
            gaps = [b["arrival_time_s"] - a["arrival_time_s"] for a, b in zip(rows, rows[1:])]
            base_gaps = [b["arrival_time_s"] - a["arrival_time_s"] for a, b in zip(base_rows, base_rows[1:])]
            self.assertEqual(gaps, base_gaps)

    def test_request_and_session_field_sets(self):
        out = list(stream_upsampled_sessions(_workload(), 2, seed="keys"))
        for session, rows in out:
            self.assertIn("session_id", session)
            self.assertIn("session_start_time_s", session)
            for row in rows:
                self.assertEqual(set(row), set(SESSION_REQUEST_KEYS))

    def test_factor_validation(self):
        for bad in (0, -2, 1.5, "3", None):
            with self.assertRaises(ValueError, msg=f"factor={bad!r}"):
                list(stream_upsampled_sessions(_workload(), bad, seed="x"))


if __name__ == "__main__":
    unittest.main()
