"""Compatibility contracts imported from the colleague's runtime interface."""

import heapq
import unittest
from types import SimpleNamespace

from astermind.engine import GlobalTimeline
from astermind.engine.cell_bridge import RuntimeCellNetwork


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




if __name__ == "__main__":
    unittest.main()
