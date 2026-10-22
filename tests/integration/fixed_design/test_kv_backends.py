"""Keep the original same-session KV behavior across every transport backend."""

from dataclasses import replace
import importlib.util
import unittest

from orbitserve.runtime import run_simulation
from tests.unit.runtime.test_runtime_engine import _request, _scenario
from tests.unit.runtime.test_runtime_event_order import strategy


class KVBackendCompatibilityTests(unittest.TestCase):
    def execute(self, mode, *, enabled=True, context=10, second_session="session-0"):
        spec = _scenario(
            replace(strategy(), kv_cache_copy_enabled=enabled),
            requests=(
                _request("first", 0),
                _request(
                    "second",
                    10,
                    lon_deg=90,
                    context_tokens_before_turn=context,
                    session_id=second_session,
                ),
            ),
        )
        result = run_simulation(
            replace(spec, network=replace(spec.network, transport_model=mode))
        )
        self.assertEqual(result.request_summary["completed_request_count"], 2)
        return result

    def verify(self, mode):
        result = self.execute(mode)
        row = result.request_records[1]
        self.assertEqual(row["decision"]["serving_compute_node_id"], "sat-b")
        self.assertEqual(row["kv"]["kv_copied_tokens"], 10)
        self.assertEqual(row["timing"]["prefill_service_s"], 0)
        copied = [
            leg for leg in row["routing"]["selected_path"] if leg["kind"] == "kv_copy"
        ]
        self.assertTrue(copied)
        self.assertTrue(all(leg["payload_bytes"] == 20 for leg in copied))
        for node in ("sat-a", "sat-b"):
            self.assertIn(
                "session-0", result.state_summary["kv_caches"][node]["entries"]
            )
        with self.subTest(case="copy_disabled"):
            row = self.execute(mode, enabled=False).request_records[1]
            self.assertEqual(row["kv"]["kv_copied_tokens"], 0)
            self.assertEqual(row["kv"]["context_recompute_tokens"], 10)
            self.assertEqual(row["timing"]["prefill_service_s"], 1)
        with self.subTest(case="session_isolation"):
            row = self.execute(mode, second_session="another-session").request_records[
                1
            ]
            self.assertEqual(row["kv"]["kv_copied_tokens"], 0)
            self.assertEqual(row["kv"]["context_recompute_tokens"], 10)
        with self.subTest(case="declared_context_cap"):
            row = self.execute(mode, context=5).request_records[1]
            self.assertEqual(row["kv"]["kv_copied_tokens"], 5)
            self.assertEqual(row["kv"]["context_recompute_tokens"], 5)
        with self.subTest(case="future_kv_is_not_visible"):
            spec = _scenario(
                strategy(),
                requests=(
                    _request("overlap-a", 0),
                    _request("overlap-b", 0.1, context_tokens_before_turn=10),
                ),
            )
            result = run_simulation(
                replace(spec, network=replace(spec.network, transport_model=mode))
            )
            self.assertEqual(result.request_summary["completed_request_count"], 2)
            self.assertEqual(result.request_records[1]["kv"]["kv_hit_tokens"], 0)

    def test_queue_preserves_kv_business_behavior(self):
        self.verify("queue")

    def test_fluid_preserves_kv_business_behavior(self):
        self.verify("fluid")

    @unittest.skipUnless(importlib.util.find_spec("simgrid"), "requires native SimGrid")
    def test_simgrid_preserves_kv_business_behavior(self):
        self.verify("simgrid")

    @unittest.skipUnless(importlib.util.find_spec("ns"), "requires native ns-3")
    def test_ns3_preserves_kv_business_behavior(self):
        self.verify("ns3")
