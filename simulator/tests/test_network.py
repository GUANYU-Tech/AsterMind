import sys
from types import SimpleNamespace
import unittest

from network_sim import CommunicationOperator, CommunicationRuntime, OperatorCommunication
from network_sim import PhysicalTopology, Scenario, TransferRequest
from network_sim.comm.ns3_runtime import DEFAULT_LIBRARY, NS3Runtime
from simulator.config import SimulationConfig
from simulator.examples.make_examples import example


class SharedNetworkTests(unittest.TestCase):
    def topology(self):
        return SimulationConfig.model_validate(example()).topology()

    def test_cell_executors_share_an_operator_comm_separate_from_transport(self):
        from simulator.engine import Simulation

        def make_cell(config, loop, on_complete, on_output, executor):
            return SimpleNamespace(executor=executor, close=lambda: None)

        config = SimulationConfig.model_validate(example())
        with Simulation(config, cell_factory=make_cell) as simulation:
            first = simulation.cells["cellA"].executor
            second = simulation.cells["cellB"].executor
            self.assertIsNot(first, second)
            self.assertIs(first.comm, second.comm)
            self.assertIs(first.comm, simulation.operator_comm)
            self.assertIsNot(first.comm, simulation.network)
            self.assertIs(first.loop, simulation.loop)
            self.assertIs(second.loop, simulation.loop)
            self.assertEqual({key[0] for key in first.placements}, {"cellA"})
            self.assertEqual({key[0] for key in second.placements}, {"cellB"})

    def test_operator_estimates_route_and_rounds_without_bandwidth_competition(self):
        raw = dict(cells=[dict(id="c", satellites=["s"])],
                   satellites=[dict(id="s")],
                   devices=[dict(id=f"g{i}", satellite="s", kind="gpu") for i in range(3)],
                   links=[])
        raw["devices"].append(dict(id="sw", satellite="s", kind="switch", capacity_gbps=4))
        for i in range(3):
            raw["links"].append(dict(id=f"l{i}", source=f"g{i}", target="sw",
                                     bandwidth_gbps=8, latency_ns=10))
        comm = OperatorCommunication(PhysicalTopology(Scenario.model_validate(raw)))
        gpu = ("g0", "g1", "g2")
        reduce = CommunicationOperator("all_reduce", "ring", gpu, 3000, "per_rank_input",
                                       ring_order=gpu, reduction="sum")
        result = comm.estimate(reduce)
        # Each round independently prices 1000 bytes on a 4 Gbps path with 20 ns latency.
        self.assertEqual((result.duration_ns, result.rounds, result.flow_count), (8080, 4, 12))
        self.assertEqual(comm.estimate(reduce), result)
        exchange = CommunicationOperator("all_to_all", "pairwise", gpu, 3500, "aggregate_input",
                                        traffic_matrix_bytes=((0, 1000, 0), (2000, 0, 0), (0, 500, 0)))
        result = comm.estimate(exchange)
        self.assertEqual((result.duration_ns, result.rounds, result.flow_count), (6040, 2, 3))

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
