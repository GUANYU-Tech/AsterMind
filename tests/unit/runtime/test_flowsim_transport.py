"""FlowSim must retain in-flight work under changing shared resources."""

import importlib.util
import unittest

from orbitserve.runtime.engine.flowsim_transport import FlowSimTransport
from orbitserve.runtime.setup.network_spec import NetworkNodeSpec


@unittest.skipUnless(importlib.util.find_spec("_flowsim"), "requires shared-runtime FlowSim")
class FlowSimTransportTests(unittest.TestCase):
    def test_sub_ulp_native_completion_drains_fractional_tail(self):
        nodes = {name: NetworkNodeSpec(name, "compute") for name in ("a", "b")}
        engine = FlowSimTransport(nodes, ({"link_id": "ab", "endpoint_a": "a", "endpoint_b": "b"},), {})
        self.addCleanup(engine.close)
        leg = {"link_id": "ab", "source_node_id": "a", "target_node_id": "b"}
        engine.advance_until(7.25)
        engine.configure({engine.resource(leg): 2.3e9})
        engine.start("tail", leg, 1_000_000.)
        sent = 0.
        for _ in range(4):
            result = engine.advance_until(8.)
            sent += result['bytes_sent'].get('tail', 0.)
            if result['completed']:
                self.assertEqual(result['completed'], ['tail'])
                break
        else:
            self.fail('native completion failed to drain the sub-ULP tail')
        self.assertAlmostEqual(sent, 1_000_000., places=6)
        self.assertAlmostEqual(result['time_s'], 7.25 + 8_000_000 / 2.3e9, places=12)

    def test_flowsim_is_default_transport(self):
        from orbitserve.runtime import LinkProfile, NetworkSpec

        nodes = (NetworkNodeSpec("a", "compute"), NetworkNodeSpec("b", "compute"))
        network = NetworkSpec(nodes, {"communication_isl": LinkProfile(1000, 0)})
        self.assertEqual(network.transport_model, "flowsim")

    def test_advance_reaches_exact_host_time_after_unit_conversion(self):
        nodes = {name: NetworkNodeSpec(name, "compute") for name in ("a", "b")}
        engine = FlowSimTransport(nodes, ({"link_id": "ab", "endpoint_a": "a", "endpoint_b": "b"},), {})
        for target in (50.003407875784525, 51.20304143987972, 60.000127969315955):
            self.assertEqual(engine.advance_until(target)["time_s"], target)
        engine.close()

    def test_capacity_change_zero_pause_and_cancel(self):
        nodes = {name: NetworkNodeSpec(name, "compute") for name in ("a", "b")}
        engine = FlowSimTransport(nodes, ({"link_id": "ab", "endpoint_a": "a", "endpoint_b": "b"},), {})
        leg = {"link_id": "ab", "source_node_id": "a", "target_node_id": "b"}
        key = engine.resource(leg)
        engine.configure({key: 8000})
        engine.start("first", leg, 1000)
        self.assertAlmostEqual(engine.advance_until(0.5)["bytes_sent"]["first"], 500)
        engine.configure({key: 0})
        self.assertEqual(engine.advance_until(1.5)["bytes_sent"]["first"], 0)
        engine.configure({key: 4000})
        result = engine.advance_until(3)
        self.assertAlmostEqual(result["time_s"], 2.5)
        self.assertAlmostEqual(result["bytes_sent"]["first"], 500)
        self.assertEqual(result["completed"], ["first"])
        engine.start("cancelled", leg, 50)
        engine.cancel(["cancelled"])
        self.assertEqual(engine.advance_until(4)["completed"], [])
        engine.close()

    def test_shared_radio_pool_across_distinct_links(self):
        nodes = {
            "g1": NetworkNodeSpec("g1", "ground"),
            "g2": NetworkNodeSpec("g2", "ground"),
            "s": NetworkNodeSpec("s", "communication",
                radio_capacity_bps={"feeder_uplink_bps": 8000, "feeder_downlink_bps": 8000},
                capacity_assumptions=("test",)),
        }
        edges = ({"link_id": "g1s", "endpoint_a": "g1", "endpoint_b": "s"},
                 {"link_id": "g2s", "endpoint_a": "g2", "endpoint_b": "s"})
        engine = FlowSimTransport(nodes, edges, {})
        legs = [{"link_id": edge["link_id"], "source_node_id": edge["endpoint_a"],
                 "target_node_id": edge["endpoint_b"]} for edge in edges]
        engine.configure({engine.resource(leg): 8000 for leg in legs})
        for index, leg in enumerate(legs):
            engine.start(str(index), leg, 1000)
        result = engine.advance_until(1)
        self.assertAlmostEqual(result["bytes_sent"]["0"], 500)
        self.assertAlmostEqual(result["bytes_sent"]["1"], 500)
        self.assertEqual(result["completed"], [])
        self.assertEqual(engine.advance_until(2)["completed"], ["0", "1"])
        engine.close()
