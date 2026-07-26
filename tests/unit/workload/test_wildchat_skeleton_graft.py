"""Unit tests for the WildChat adapter skeleton graft (``--skeleton-bundle``).

Split from test_wildchat_diurnal_profile.py: covers only the graft path —
slot preservation, same-country content matching with global fallback,
overspan dropping, determinism, warp/graft mutex, and an end-to-end mini build.
Synthetic in-memory rows; no Parquet/tokenizer needed.
"""

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


ADAPTER_PATH = Path(__file__).resolve().parents[3] / "src" / "workloads" / "wildchat" / "wildchat_adapter.py"
_spec = importlib.util.spec_from_file_location("wildchat_adapter", ADAPTER_PATH)
wa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wa)


EPOCH = datetime(2025, 3, 1, tzinfo=timezone.utc)
BIN_S = 300


def _row(country, offset_s, *, epoch=EPOCH, turns=2, gap_s=30.0, state=None):
    conversation = []
    t = epoch + timedelta(seconds=offset_s)
    for index in range(turns):
        user = {"role": "user", "content": f"question {index}", "country": country}
        if state is not None:
            user["state"] = state
        conversation.append(user)
        conversation.append({
            "role": "assistant",
            "content": f"answer {index}",
            "timestamp": (t + timedelta(seconds=1)).isoformat(),
        })
        t += timedelta(seconds=gap_s)
    return {"conversation": conversation, "model": "m"}


def _region(region_id):
    return {
        "region_id": region_id,
        "region_name": region_id,
        "lat_deg": 0.0,
        "lon_deg": 0.0,
        "timezone_utc_offset_h": 0,
        "market_class": None,
    }


def _config(workload_id="graft_test"):
    return {
        "workload_id": workload_id,
        "seed": 7,
        "dataset": {
            "name": "synthetic",
            "revision": "rev",
            "time_window": {
                "time_mode": "month_compressed_day",
                "epoch_utc": "2025-03-01T00:00:00Z",
                "window_end_utc": "2025-04-01T00:00:00Z",
                "time_horizon_s": 2678400,
                "output_epoch_utc": "2025-03-15T00:00:00Z",
                "output_time_horizon_s": 86400,
                "time_bin_s": BIN_S,
            },
        },
        "region_mapping": {
            "United States|*": _region("us"),
            "China|*": _region("cn"),
            "Germany|*": _region("de"),
            "Russia|*": _region("ru"),
        },
        "adapter_policy": {
            "arrival_time_policy": "assistant_completion_proxy",
            "unknown_region_policy": "drop",
            "missing_timestamp_policy": "drop",
            "invalid_pair_policy": "drop",
            "record_filter": {"require_nonempty_content": True},
        },
        "diurnal_profile": {
            "grouping": "macro_region",
            "epoch_utc": "2023-05-01T00:00:00Z",
            "window_end_utc": "2023-06-01T00:00:00Z",
            "expected_source_sha256": {},
        },
    }


def _slot(slot_session_id, start, region_id, region_name, lat=55.75, lon=37.61):
    region = {
        "region_id": region_id,
        "region_name": region_name,
        "lat_deg": lat,
        "lon_deg": lon,
        "timezone_utc_offset_h": 3,
        "market_class": None,
    }
    return {
        "slot_session_id": slot_session_id,
        "region_id": region_id,
        "start": float(start),
        "region": region,
        "country": wa.slot_country(region_name),
    }


def _graft_config(workload_id="graft_test"):
    config = _config(workload_id)
    return config


def _graft_pool(config, rows):
    return wa.collect_graft_content_pool(config, rows, "abc123")


def _graft_signature(extracted):
    return [
        (session_id, key, [(p["arrival_time_s"], p["region"]["region_id"], p["pair_index"]) for p in pairs])
        for session_id, key, _raw, pairs in extracted
    ]


class SkeletonGraftTests(unittest.TestCase):
    def test_slot_country_parsing_and_aliases(self):
        self.assertEqual(wa.slot_country("Moscow Oblast, Russia"), "Russia")
        self.assertEqual(wa.slot_country("South Korea"), "South Korea")
        self.assertEqual(wa.slot_country("New South Wales, Australia"), "Australia")
        pool = {"The Netherlands": [1], "United States": [2]}
        self.assertEqual(wa._content_pool_key("Netherlands", pool), "The Netherlands")
        self.assertEqual(wa._content_pool_key("United States of America", pool), "United States")
        self.assertIsNone(wa._content_pool_key("Freedonia", pool))

    def test_slot_preservation_and_same_country_priority(self):
        config = _graft_config("graft_prio_10x")
        slots = [
            _slot("skel_a", 100.0, "russia_moscow", "Moscow, Russia"),
            _slot("skel_b", 200.0, "us_california", "California, United States", lat=36.7, lon=-119.4),
        ]
        rows = [
            _row("Russia", 10),
            _row("United States", 20),
        ]
        pool, global_pool = _graft_pool(config, rows)
        extracted, info, warnings = wa.plan_skeleton_graft(config, slots, pool, global_pool, "10x")
        self.assertEqual(warnings, [])
        self.assertEqual(len(extracted), 2)
        for (session_id, _, raw, pairs), slot in zip(extracted, slots):
            # 槽位保持：首轮到达时刻与区域与骨架完全一致
            self.assertEqual(pairs[0]["arrival_time_s"], slot["start"])
            self.assertTrue(all(p["region"] is slot["region"] for p in pairs))
        # 同国优先：槽位 a 用俄罗斯内容，槽位 b 用美国内容
        self.assertEqual(extracted[0][2]["conversation"][0]["country"], "Russia")
        self.assertEqual(extracted[1][2]["conversation"][0]["country"], "United States")
        self.assertEqual(info["countries"]["Russia"]["slots"], 1)
        self.assertEqual(info["countries"]["Russia"]["fallback_slots"], 0)
        self.assertEqual(info["countries"]["United States"]["slots"], 1)

    def test_global_fallback_when_country_pool_empty(self):
        config = _graft_config("graft_fallback_10x")
        slots = [_slot("skel_jp", 50.0, "country_japan", "Japan", lat=35.6, lon=139.6)]
        rows = [_row(None, 10), _row("Russia", 20)]  # null-country 行进全球池
        pool, global_pool = _graft_pool(config, rows)
        self.assertEqual(len(global_pool), 1)
        extracted, info, _ = wa.plan_skeleton_graft(config, slots, pool, global_pool, "10x")
        self.assertEqual(len(extracted), 1)
        self.assertIsNone(extracted[0][2]["conversation"][0]["country"])  # 用了全球池的 null-country 内容
        self.assertEqual(info["countries"]["Japan"]["fallback_slots"], 1)

    def test_overspan_content_is_dropped(self):
        config = _graft_config("graft_drop_10x")
        slots = [
            _slot("skel_late", 86000.0, "russia_moscow", "Moscow, Russia"),   # 86000+span 越界
            _slot("skel_ok", 100.0, "russia_moscow", "Moscow, Russia"),
        ]
        rows = [_row("Russia", 10, turns=2, gap_s=5000.0), _row(None, 20)]
        pool, global_pool = _graft_pool(config, rows)
        extracted, info, warnings = wa.plan_skeleton_graft(config, slots, pool, global_pool, "10x")
        self.assertEqual(len(extracted), 1)
        self.assertEqual(info["dropped_sessions"], 1)
        self.assertTrue(any("dropped 1 session(s)" in w for w in warnings))
        # 超长跨度（>24h）内容本身：slot 早也无法容纳
        slots2 = [_slot("skel_x", 100.0, "russia_moscow", "Moscow, Russia")]
        rows2 = [_row("Russia", 10, turns=2, gap_s=90000.0)]
        pool2, global2 = _graft_pool(config, rows2)
        extracted2, info2, _ = wa.plan_skeleton_graft(config, slots2, pool2, global2, "10x")
        self.assertEqual(extracted2, [])
        self.assertEqual(info2["dropped_sessions"], 1)

    def test_graft_deterministic_across_runs(self):
        config = _graft_config("graft_det_5x")
        slots = [_slot(f"skel_{i:03d}", float(i * 100), "russia_moscow", "Moscow, Russia") for i in range(30)]
        slots += [_slot(f"skel_us_{i:03d}", float(i * 130 + 7), "us_california", "California, United States") for i in range(10)]
        rows = [_row("Russia", i * 4000 + 3, turns=2) for i in range(8)]
        rows += [_row("United States", i * 5000 + 11, turns=2) for i in range(3)]
        rows += [_row(None, 99)]
        pool, global_pool = _graft_pool(config, rows)
        first = wa.plan_skeleton_graft(config, slots, pool, global_pool, "5x")
        second = wa.plan_skeleton_graft(config, slots, pool, global_pool, "5x")
        self.assertEqual(_graft_signature(first[0]), _graft_signature(second[0]))
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[1]["slot_count_used"], 20)  # 40 槽位 × 0.5

    def test_graft_conflicts_with_profile_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config("graft_mutex")
            config.update({
                "region_mapping": {"Russia|*": _region("ru")},
                "gateways": {"selection_policy": "nearest_gateway", "items": []},
                "tokenization": {"backend": "simple_unicode_v1", "target_model": "t", "context_window_tokens": 16,
                                 "bytes_per_input_token": 4, "bytes_per_output_token": 4},
                "model_mapping": {"default": "t"},
                "slo_policy": {"session_assignment": "max_output_tokens", "rules": [
                    {"default": True, "slo_class": "normal_long", "ttft_deadline_s": 1.0,
                     "e2e_deadline_s": 2.0, "priority": "normal", "session_ttl_s": 600.0}]},
                "bundle_output": {"module_version": "1.0.0", "float_precision_digits": 6, "overwrite": True},
                "country_centroids_path": None,
                "state_centroids_path": None,
                "skeleton_bundle": {},
            })
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(wa.AdapterError) as ctx:
                wa.build(config_path, [Path(tmp) / "in.parquet"], Path(tmp) / "out", True,
                         profile_paths=[Path(tmp) / "p.parquet"], skeleton_bundle=Path(tmp) / "skel")
            self.assertIn("conflicts", str(ctx.exception))

    def test_graft_build_end_to_end_slot_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # 迷你骨架 bundle：4 个槽位、2 个区域
            skeleton_dir = tmp_path / "skeleton"
            skeleton_dir.mkdir()
            regions = [
                {"region_id": "russia_moscow", "region_name": "Moscow, Russia", "lat_deg": 55.75, "lon_deg": 37.61,
                 "timezone_utc_offset_h": 3.0, "demand_weight": 0.5, "weight_source": "t", "market_class": None},
                {"region_id": "country_freedonia", "region_name": "Freedonia", "lat_deg": 1.0, "lon_deg": 2.0,
                 "timezone_utc_offset_h": 0.0, "demand_weight": 0.5, "weight_source": "t", "market_class": None},
            ]
            (skeleton_dir / "regions.json").write_text(json.dumps(regions), encoding="utf-8")
            slot_rows = [
                {"session_id": "skel_001", "user_region_id": "russia_moscow", "session_start_time_s": 100.0,
                 "session_turn_count": 1, "session_slo_class": "x", "context_window_tokens": 16, "session_ttl_s": 600.0, "seed": 1},
                {"session_id": "skel_002", "user_region_id": "russia_moscow", "session_start_time_s": 200.0,
                 "session_turn_count": 1, "session_slo_class": "x", "context_window_tokens": 16, "session_ttl_s": 600.0, "seed": 1},
                {"session_id": "skel_003", "user_region_id": "country_freedonia", "session_start_time_s": 300.0,
                 "session_turn_count": 1, "session_slo_class": "x", "context_window_tokens": 16, "session_ttl_s": 600.0, "seed": 1},
                {"session_id": "skel_004", "user_region_id": "country_freedonia", "session_start_time_s": 400.0,
                 "session_turn_count": 1, "session_slo_class": "x", "context_window_tokens": 16, "session_ttl_s": 600.0, "seed": 1},
            ]
            (skeleton_dir / "sessions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in slot_rows), encoding="utf-8")
            config = _config("graft_e2e")
            config.update({
                "region_mapping": {"Russia|*": _region("ru")},
                "gateways": {"selection_policy": "nearest_gateway", "items": [
                    {"gateway_id": "gw", "lat_deg": 0.0, "lon_deg": 0.0, "gateway_bandwidth_bps": 1,
                     "availability_profile_id": None, "cost_usd_per_year": None}]},
                "tokenization": {"backend": "simple_unicode_v1", "target_model": "t", "context_window_tokens": 1024,
                                 "bytes_per_input_token": 4, "bytes_per_output_token": 4},
                "model_mapping": {"default": "t"},
                "slo_policy": {"session_assignment": "max_output_tokens", "rules": [
                    {"default": True, "slo_class": "normal_long", "ttft_deadline_s": 1.0,
                     "e2e_deadline_s": 2.0, "priority": "normal", "session_ttl_s": 600.0}]},
                "bundle_output": {"module_version": "1.0.0", "float_precision_digits": 6, "overwrite": True},
                "country_centroids_path": None,
                "state_centroids_path": None,
                "skeleton_bundle": {"regions_file": "regions.json", "sessions_file": "sessions.jsonl"},
            })
            config["dataset"]["time_window"] = {
                "time_mode": "month_compressed_day",
                "epoch_utc": "2025-03-01T00:00:00Z",
                "window_end_utc": "2025-04-01T00:00:00Z",
                "time_horizon_s": 2678400,
                "output_epoch_utc": "2023-05-15T00:00:00Z",
                "output_time_horizon_s": 86400,
                "time_bin_s": 300,
            }
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            dummy = tmp_path / "input.parquet"
            dummy.write_bytes(b"fake content")
            rows = [_row("Russia", 10, turns=2), _row(None, 20, turns=2)]
            with mock.patch.object(wa, "read_source_rows", return_value=list(rows)):
                wa.build(config_path, [dummy], tmp_path / "out" / "experiment_bundle_10x", True,
                         scale_flags={"10x": True}, skeleton_bundle=skeleton_dir)
            out_dir = tmp_path / "out" / "experiment_bundle_10x"
            sessions = [json.loads(line) for line in (out_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines()]
            requests = [json.loads(line) for line in (out_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
            # 槽位保持：4 个槽位全部嫁接，开始时刻与骨架完全一致
            self.assertEqual(
                sorted(s["session_start_time_s"] for s in sessions),
                [100.0, 200.0, 300.0, 400.0],
            )
            self.assertEqual(
                sorted(s["user_region_id"] for s in sessions),
                ["country_freedonia", "country_freedonia", "russia_moscow", "russia_moscow"],
            )
            # 每个请求都在其槽位开始时刻之后不久（内容跨度 30s）
            for row in requests:
                self.assertTrue(any(abs(row["arrival_time_s"] - s) <= 31.0 for s in (100.0, 200.0, 300.0, 400.0)))
            manifest = json.loads((out_dir / "workload_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["time"]["time_mode"], "skeleton_graft")
            graft = manifest["provenance"]["skeleton_graft"]
            self.assertEqual(graft["grafted_sessions"], 4)
            self.assertEqual(graft["countries"]["Freedonia"]["fallback_slots"], 2)
            self.assertEqual(graft["countries"]["Russia"]["slots"], 2)


if __name__ == "__main__":
    unittest.main()
