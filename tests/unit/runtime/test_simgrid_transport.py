"""Native SimGrid regressions with analytical answers, not mocked durations."""

import importlib.util
import unittest
from dataclasses import replace
from unittest.mock import patch

from astermind.engine import LinkProfile, NetworkSpec, check_network_backend
from astermind.engine.simgrid_transport import SimGridTransport, SimGridUnavailable
from tests.network_case import NetworkCase, run_traffic, traffic


def transport():
    return SimGridTransport(
        ("a", "b", "c"),
        (
            {"link_id": "ab", "endpoint_a": "a", "endpoint_b": "b", "kind": "isl"},
            {"link_id": "bc", "endpoint_a": "b", "endpoint_b": "c", "kind": "isl"},
        ),
        {"isl": LinkProfile(8.0, 0.0)},
    )


def leg(source="a", target="b", link="ab"):
    return {"link_id": link, "source_node_id": source, "target_node_id": target}


class TransportConfigurationTests(unittest.TestCase):
    def test_default_is_queue_and_unknown_model_is_rejected(self):
        scenario = NetworkCase(traffic=(traffic(),))
        self.assertEqual(scenario.network.transport_model, "queue")
        self.assertEqual(
            NetworkSpec.from_value(scenario.network.to_dict()), scenario.network
        )
        with self.assertRaisesRegex(ValueError, "transport_model"):
            replace(scenario.network, transport_model="simgrid-like")

    def test_explicit_missing_simgrid_fails_without_changing_queue_mode(self):
        scenario = NetworkCase(traffic=(traffic(),))
        selected = replace(
            scenario, network=replace(scenario.network, transport_model="simgrid")
        )
        with patch(
            'astermind.engine.simgrid_transport.importlib.util.find_spec',
            return_value=None,
        ):
            with self.assertRaisesRegex(SimGridUnavailable, "optional dependency"):
                check_network_backend(selected.network.transport_model)
            self.assertEqual(
                run_traffic(scenario).completed, 1
            )


@unittest.skipUnless(
    importlib.util.find_spec("simgrid"),
    "Install OrbitServe[simgrid] to run native backend tests",
)
class NativeSimGridTransportTests(unittest.TestCase):
    def setUp(self):
        self.backend = transport()
        self.addCleanup(self.backend.close)

    def test_equal_flows_share_link_and_finish_together(self):
        self.backend.start("a", leg(), 1)
        self.backend.start("b", leg(), 1)
        result = self.backend.advance_until(10)
        self.assertAlmostEqual(result["time_s"], 2.0, places=9)
        self.assertEqual(result["completed"], ["a", "b"])
        self.assertAlmostEqual(result["bytes_sent"]["a"], 1.0)
        self.assertAlmostEqual(result["bytes_sent"]["b"], 1.0)
        self.assertEqual(
            self.backend.provenance["completion_source"], "native_simgrid_activity"
        )

    def test_later_flow_changes_existing_completion_and_releases_bandwidth(self):
        self.backend.start("a", leg(), 1)
        initial = self.backend.advance_until(0.5)
        self.assertAlmostEqual(initial["bytes_sent"]["a"], 0.5)
        self.backend.start("b", leg(), 1)
        first = self.backend.advance_until(10)
        self.assertEqual(first["completed"], ["a"])
        self.assertAlmostEqual(first["time_s"], 1.5, places=9)
        second = self.backend.advance_until(10)
        self.assertEqual(second["completed"], ["b"])
        self.assertAlmostEqual(second["time_s"], 2.0, places=9)

    def test_opposite_directions_and_independent_links_do_not_contend(self):
        self.backend.start("forward", leg(), 1)
        self.backend.start("reverse", leg("b", "a"), 1)
        self.backend.start("independent", leg("b", "c", "bc"), 1)
        result = self.backend.advance_until(10)
        self.assertAlmostEqual(result["time_s"], 1.0, places=9)
        self.assertEqual(len(result["completed"]), 3)

    def test_capacity_updates_preserve_native_remaining_bytes(self):
        self.backend.start("a", leg(), 1)
        first = self.backend.advance_until(0.5)
        self.backend.configure({self.backend.resource(leg()): 4})
        last = self.backend.advance_until(10)
        self.assertAlmostEqual(last["time_s"], 1.5, places=9)
        self.assertAlmostEqual(first["bytes_sent"]["a"] + last["bytes_sent"]["a"], 1.0)

    def test_zero_capacity_suspends_and_resumes_without_phantom_bytes(self):
        self.backend.start("a", leg(), 1)
        self.backend.advance_until(0.25)
        key = self.backend.resource(leg())
        self.backend.configure({key: 0})
        paused = self.backend.advance_until(2)
        self.assertEqual(paused["completed"], [])
        self.assertEqual(paused["bytes_sent"]["a"], 0)
        self.backend.configure({key: 8})
        last = self.backend.advance_until(10)
        self.assertAlmostEqual(last["time_s"], 2.75, places=9)

    def test_cancel_releases_bandwidth_and_child_engine_is_reaped(self):
        self.backend.start("a", leg(), 1)
        self.backend.start("b", leg(), 1)
        self.backend.advance_until(0.5)
        self.backend.cancel(["a"])
        last = self.backend.advance_until(10)
        self.assertEqual(last["completed"], ["b"])
        self.assertAlmostEqual(last["time_s"], 1.25, places=9)
        self.backend.close()
        self.assertIsNotNone(self.backend.process.poll())

    def test_fractional_analytical_payload_is_not_rounded_to_whole_bytes(self):
        self.backend.start("token", leg(), 1 / 3)
        last = self.backend.advance_until(10)
        self.assertAlmostEqual(last["time_s"], 1 / 3, places=6)
