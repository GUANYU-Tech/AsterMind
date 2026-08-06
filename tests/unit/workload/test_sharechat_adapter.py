"""Regression coverage for the ShareChat multi-turn bundle adapter."""

import importlib.util
import base64
import json
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = ROOT / "src" / "workloads" / "sharechat" / "sharechat_adapter.py"
GLOBE_EXPORT_PATH = ROOT / "src" / "workloads" / "sharechat" / "export_globe_bundle.py"
FIXTURE = ROOT / "src" / "workloads" / "sharechat" / "fixtures" / "sharechat_timed_fixture.csv"
spec = importlib.util.spec_from_file_location("sharechat_adapter", ADAPTER_PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
globe_spec = importlib.util.spec_from_file_location("sharechat_globe_export", GLOBE_EXPORT_PATH)
globe_export = importlib.util.module_from_spec(globe_spec)
globe_spec.loader.exec_module(globe_export)


class ShareChatAdapterTests(unittest.TestCase):
    def test_globe_export_compacts_visual_fields_only(self):
        payload, count, horizon = globe_export.pack_rows([
            {"arrival_time_s": 1.6, "lat_deg": 39.9042, "lon_deg": 116.4074, "region_id": "country_china"},
            {"arrival_time_s": 12.2, "lat_deg": 51.5074, "lon_deg": -0.1278, "region_id": "country_united_kingdom"},
        ])
        decoded = base64.b64decode(payload)
        self.assertEqual((count, horizon, len(decoded)), (2, 12, 18))
        self.assertEqual(struct.unpack("<IhhB", decoded[:9]), (2, 3990, 11641, 2))
        self.assertEqual(struct.unpack("<IhhB", decoded[9:]), (12, 5151, -13, 3))

    def test_timed_multiturn_session_is_preserved_at_second_resolution(self):
        # Keep the deterministic test output under the writable workspace;
        # the desktop sandbox blocks temporary-directory ACL changes.
        out = ROOT / "src" / "workloads" / "sharechat" / "bundles" / "_unit_test_1s" / "experiment_bundle"
        args = SimpleNamespace(
                input=str(FIXTURE), out=str(out), workload_id="sharechat_test",
                countries=str(adapter.DEFAULT_COUNTRIES), profile=str(adapter.DEFAULT_PROFILE),
                profile_platform="ChatGPT", epoch="2026-08-03T00:00:00Z", max_sessions=2,
                seed=11, time_bin_s=1, context_window_tokens=131072, session_ttl_s=600.0,
                require_turn_timestamps=True, overwrite=True,
                tokenizer_backend="simple_unicode_v1", tokenizer_id="smoke", tokenizer_local_dir="unused",
                compute_source_sha256=False,
        )
        adapter.build(args)
        sessions = [json.loads(line) for line in (out / "sessions.jsonl").read_text(encoding="utf-8").splitlines()]
        requests = [json.loads(line) for line in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(sessions), 2)
        self.assertEqual(len(requests), 5)
        by_session = {}
        for request in requests:
            by_session.setdefault(request["session_id"], []).append(request)
        self.assertEqual(sorted(len(rows) for rows in by_session.values()), [2, 3])
        for session_id, rows in by_session.items():
            rows.sort(key=lambda row: row["turn_index"])
            self.assertEqual(rows[0]["turn_index"], 0)
            self.assertIsNone(rows[0]["inter_turn_gap_s"])
            self.assertTrue(all(row["session_id"] == session_id for row in rows))
            self.assertTrue(all(rows[index]["arrival_time_s"] > rows[index - 1]["arrival_time_s"] for index in range(1, len(rows))))
            self.assertTrue(all(rows[index]["inter_turn_gap_s"] > 0 for index in range(1, len(rows))))
            self.assertTrue(all(rows[index]["context_tokens_before_turn"] > rows[index - 1]["context_tokens_before_turn"] for index in range(1, len(rows))))
        manifest = json.loads((out / "workload_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["time"]["time_bin_s"], 1)
        self.assertEqual(manifest["time"]["time_mode"], "country_local_second_resolution")


if __name__ == "__main__":
    unittest.main()
