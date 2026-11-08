"""Compatibility contracts imported from the colleague's runtime interface."""

import contextlib
import heapq
import json
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from orbitserve.runtime import GlobalTimeline
from orbitserve.runtime.communication import FrameRouter
from orbitserve.runtime.engine import ns3_worker
from orbitserve.runtime.engine.cell_bridge import RuntimeCellNetwork
from orbitserve.runtime.engine.ns3_transport import Ns3Transport, Ns3Unavailable
from orbitserve.runtime.network import (
    CandidateTopologyGraph,
    CandidateTopologyLink,
    EstablishedLink,
    LinkState,
    NetworkNodeState,
    NetworkStateFrame,
)


class CellContractTests(unittest.TestCase):
    def test_native_gpu_alias_cannot_cross_cell_or_shadow_physical_nodes(self):
        host = SimpleNamespace(
            net=SimpleNamespace(nodes={"a": object(), "b": object()})
        )
        network = RuntimeCellNetwork(host, cell_members={"a": "A", "b": "B"})
        network.bind_endpoint("gpu0", "a", "A")
        self.assertEqual(network.resolve("gpu0"), "a")
        with self.assertRaises(ValueError):
            network.bind_endpoint("gpu1", "b", "A")
        with self.assertRaises(ValueError):
            network.bind_endpoint("gpu0", "b", "B")
        with self.assertRaises(ValueError):
            network.bind_endpoint("b", "a", "A")

    def test_standalone_timeline_orders_events_and_rejects_past(self):
        loop = GlobalTimeline()
        calls = []
        loop.schedule(1, 2, "second", lambda: calls.append(2))
        loop.schedule(1, 1, "first", lambda: calls.append(1))
        loop.run(until_s=2)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(loop.now_s, 2)
        with self.assertRaises(ValueError):
            loop.schedule(1, 0, "past", lambda: None)

    def test_hosted_timeline_uses_host_clock_queue_and_priority(self):
        queue = []
        host = SimpleNamespace(now=4.0)
        host.push = lambda time, priority, kind, payload: heapq.heappush(
            queue, (time, priority, kind, payload)
        )
        loop = GlobalTimeline(host=host)
        calls = []
        loop.schedule(5, 3, "dispatch", lambda: calls.append(loop.now_s))
        self.assertEqual(loop.pending_count, 1)
        with self.assertRaisesRegex(RuntimeError, "second Cell"):
            loop.run()
        host.now, priority, _, callback = heapq.heappop(queue)
        self.assertEqual(priority, 33)
        callback()
        self.assertEqual(calls, [5])
        self.assertEqual(loop.pending_count, 0)

    def test_frame_router_preserves_directional_rates_and_rf_pools(self):
        candidate = CandidateTopologyLink("ab", "a", "b", "rf")
        link = EstablishedLink(
            "ab",
            "a",
            "b",
            "sgl",
            100,
            0,
            "rf",
            attributes={
                "directions": [
                    {
                        "source_node_id": "a",
                        "target_node_id": "b",
                        "bandwidth_bps": 100,
                        "shared_pools": ["uplink"],
                    },
                    {
                        "source_node_id": "b",
                        "target_node_id": "a",
                        "bandwidth_bps": 20,
                        "shared_pools": ["downlink"],
                    },
                ]
            },
        )
        frame = NetworkStateFrame(
            0,
            None,
            {n: NetworkNodeState(n, n) for n in ("a", "b")},
            CandidateTopologyGraph(("a", "b"), (candidate,)),
            (LinkState(candidate, "established", "", link),),
        )
        router = FrameRouter()
        self.assertEqual(router.route(frame, "a", "b").hops[0].bandwidth_bps, 100)
        reverse = router.route(frame, "b", "a").hops[0]
        self.assertEqual(reverse.bandwidth_bps, 20)
        self.assertEqual(reverse.shared_capacity_pools, ("downlink",))

    def test_ns3_worker_keeps_trace_counters_and_explicit_capabilities(self):
        engine = MagicMock()
        engine.value.return_value = 0.0
        engine.completed.return_value = []
        engine.lib.os_packetCount.return_value = 0
        commands = [
            {"links": []},
            {"op": "trace", "enabled": True},
            {"op": "advance", "until_s": 1},
            {"op": "close"},
        ]
        output = StringIO()
        with (
            patch.object(
                ns3_worker.sys,
                "stdin",
                StringIO("\n".join(map(json.dumps, commands)) + "\n"),
            ),
            patch.object(ns3_worker, "version", return_value="test"),
            contextlib.redirect_stdout(output),
        ):
            ns3_worker.serve(engine)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        transport = Ns3Transport.__new__(Ns3Transport)
        transport.validate_ready(rows[0])
        self.assertTrue(rows[1]["tracing"])
        self.assertEqual(rows[2]["packets"], [])
        self.assertEqual(rows[2]["native_counters"]["packets_started"], 0)
        with self.assertRaisesRegex(Ns3Unavailable, "incompatible"):
            transport.validate_ready({"ready": True, "version": "old"})


if __name__ == "__main__":
    unittest.main()
