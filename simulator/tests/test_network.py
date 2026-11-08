import sys
from types import SimpleNamespace
import unittest

from network_sim import CommunicationRuntime, TransferRequest
from network_sim.comm.ns3_runtime import DEFAULT_LIBRARY, NS3Runtime
from simulator.config import SimulationConfig
from simulator.examples.make_examples import example


class SharedNetworkTests(unittest.TestCase):
    def topology(self):
        return SimulationConfig.model_validate(example()).topology()

    def test_cell_executors_are_distinct_and_share_the_network_driver(self):
        from simulator.engine import Simulation

        def make_cell(config, loop, on_complete, on_output, executor):
            return SimpleNamespace(executor=executor, close=lambda: None)

        config = SimulationConfig.model_validate(example())
        with Simulation(config, cell_factory=make_cell) as simulation:
            first = simulation.cells["cellA"].executor
            second = simulation.cells["cellB"].executor
            self.assertIsNot(first, second)
            self.assertIs(first.transport, second.transport)
            self.assertIs(first.transport, simulation.transport)
            self.assertIs(first.transport.network, simulation.network)
            self.assertIs(first.loop, simulation.loop)
            self.assertIs(second.loop, simulation.loop)
            self.assertEqual({key[0] for key in first.placements}, {"cellA"})
            self.assertEqual({key[0] for key in second.placements}, {"cellB"})

    def test_three_origins_contend_on_one_resource(self):
        with CommunicationRuntime(self.topology()) as runtime:
            for i, (origin, operation) in enumerate((("orbitserve", "request_input"), ("midlayer", "kv_migration"), ("multi_gpu", "all_to_all"))):
                runtime.submit(TransferRequest(id=f"t{i}", source="user", target="gpu0",
                    origin=origin, operation=operation, size_bytes=1000000))
            events = runtime.run()
            self.assertEqual(len(events), 3)
            self.assertEqual(len({e.completion_time_ns for e in events}), 1)
            self.assertGreater(events[0].completion_time_ns, 24_000_000)
            self.assertTrue(all(e.path[0] == "user" and e.path[-1] == "gpu0" for e in events))

    @unittest.skipUnless(sys.platform != "win32" and DEFAULT_LIBRARY.is_file(), "Linux ns-3 build required")
    def test_real_ns3_packets_and_fifo_competition(self):
        with NS3Runtime(self.topology(), packet_bytes=1000) as runtime:
            runtime.submit(TransferRequest(id="one", source="user", target="gpu0", size_bytes=10000))
            runtime.submit(TransferRequest(id="two", source="user", target="gpu0", size_bytes=10000))
            events = runtime.run()
            self.assertEqual(len(events), 2)
            self.assertTrue(all(e.kind == "completed" for e in events))
            self.assertLess(events[0].completion_time_ns, events[1].completion_time_ns)
            self.assertGreater(events[1].completion_time_ns, 160000)
            self.assertGreaterEqual(runtime.snapshot()["packet_hop_transmissions"], 60)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.snapshot()


if __name__ == "__main__":
    unittest.main()
