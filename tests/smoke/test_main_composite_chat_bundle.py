"""Smoke test: verify the refactored canonical bundle format loads correctly."""
import json
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orbitserve.workload import load_workload_bundle
from orbitserve.workload.contract import (
    SESSION_REQUEST_KEYS, REGION_KEYS, GATEWAY_KEYS, TIMEBIN_FIELDS,
    MANIFEST_FILE,
)

BUNDLES_DIR = Path(__file__).resolve().parents[2] / "src" / "workloads" / "main_composite_chat" / "bundles"
PATTERNS = ["work_weekday", "work_weekend", "nonwork_weekday", "nonwork_weekend"]


class BundleFormatTest(unittest.TestCase):

    def _bundle_dir(self, pattern: str) -> Path:
        return BUNDLES_DIR / f"chatgpt_{pattern}"

    # ── File count ──────────────────────────────────────────────────
    def test_exactly_six_files(self):
        for p in PATTERNS:
            d = self._bundle_dir(p)
            files = sorted(f.name for f in d.iterdir() if f.is_file())
            self.assertEqual(len(files), 6, f"{p}: expected 6 files, got {files}")

    def test_no_sessions_file(self):
        for p in PATTERNS:
            d = self._bundle_dir(p)
            self.assertFalse((d / "sessions.jsonl").exists(), f"{p}: sessions.jsonl should not exist")
            self.assertTrue((d / "session_requests.jsonl").exists(), f"{p}: missing session_requests.jsonl")

    # ── Manifest ────────────────────────────────────────────────────
    def test_manifest_format(self):
        for p in PATTERNS:
            d = self._bundle_dir(p)
            m = json.loads((d / MANIFEST_FILE).read_text(encoding="utf-8"))
            self.assertEqual(m["schema_version"], "orbitserve.workload")
            self.assertEqual(m["canonicalization"], "orbitserve-json-c14n")
            self.assertTrue(m["workload_id"].startswith("main_composite_chat_"))
            self.assertIn("bundle_sha256", m)
            self.assertIn("seed", m)
            self.assertIn("time_horizon_s", m)
            # Ensure old fields are gone
            for bad in ("files", "counts", "warnings", "generator", "time"):
                self.assertNotIn(bad, m, f"{p}: manifest should not contain '{bad}'")

    # ── Region / gateway keys ───────────────────────────────────────
    def test_region_keys(self):
        for p in PATTERNS:
            d = self._bundle_dir(p)
            regions = json.loads((d / "regions.json").read_text(encoding="utf-8"))
            for r in regions:
                self.assertEqual(set(r.keys()), set(REGION_KEYS), f"{p}: region keys mismatch")

    def test_gateway_keys(self):
        for p in PATTERNS:
            d = self._bundle_dir(p)
            gateways = json.loads((d / "gateways.json").read_text(encoding="utf-8"))
            for g in gateways:
                self.assertEqual(set(g.keys()), set(GATEWAY_KEYS), f"{p}: gateway keys mismatch")

    # ── Session request keys ────────────────────────────────────────
    def test_session_request_keys(self):
        for p in PATTERNS:
            d = self._bundle_dir(p)
            with open(d / "session_requests.jsonl", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i > 50:
                        break
                    row = json.loads(line)
                    self.assertEqual(set(row.keys()), set(SESSION_REQUEST_KEYS),
                                     f"{p}: session_request[{i}] keys mismatch")

    # ── Timebin ─────────────────────────────────────────────────────
    def test_timebin_granularity(self):
        for p in PATTERNS:
            d = self._bundle_dir(p)
            with open(d / "timebin_demand.csv", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                self.assertEqual(header, list(TIMEBIN_FIELDS), f"{p}: timebin header mismatch")
                for line in f:
                    parts = line.strip().split(",")
                    tb_start = float(parts[0])
                    tb_end = float(parts[1])
                    self.assertEqual(tb_end - tb_start, 60.0,
                                     f"{p}: timebin should be 60s, got {tb_end - tb_start}s")
                    break

    # ── Summary ─────────────────────────────────────────────────────
    def test_summary_p50_p95(self):
        required = {
            "workload_id", "request_count", "session_count",
            "multi_turn_session_count",
            "session_turn_count_p50", "session_turn_count_p95",
            "request_prompt_tokens_p50", "request_prompt_tokens_p95",
            "request_output_tokens_p50", "request_output_tokens_p95",
            "session_tokens_p50", "session_tokens_p95",
            "session_duration_s_p50", "session_duration_s_p95",
        }
        for p in PATTERNS:
            d = self._bundle_dir(p)
            s = json.loads((d / "workload_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(s.keys()), required, f"{p}: summary keys mismatch")

    # ── Full loader round-trip ──────────────────────────────────────
    def test_load_workload_bundle(self):
        for p in PATTERNS:
            result = load_workload_bundle(str(self._bundle_dir(p)))
            self.assertEqual(result.status, "loaded")
            self.assertGreater(len(result.session_requests), 0)
            self.assertGreater(len(result.regions), 0)
            self.assertGreater(len(result.gateways), 0)
            # Check model_family is from platform (not hardcoded)
            models = {r["model_family"] for r in result.session_requests[:100]}
            self.assertTrue(models, f"{p}: model_family should not be empty")


if __name__ == "__main__":
    unittest.main()
